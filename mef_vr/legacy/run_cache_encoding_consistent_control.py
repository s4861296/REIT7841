
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE = Path('/mnt/d/VRFlickerLab')
EVIDENCE = BASE / 'model_evaluation/r3d18_flicker_v1/motion_fusion_native_evidence_20260921_v1'
TEACHER = BASE / 'model_training/r3d18_flicker_v1/factorial100_normal_consistency_20260921_v1/without_consistency'
C_TRIAL = BASE / 'model_training/r3d18_flicker_v1/native_motion_input_F100_pairing_recovery_20260925_v3/trial'
GAP_AUDIT = BASE / 'model_evaluation/r3d18_flicker_v1/cache_train_short_gap_input_audit_20261004_v1'
MODES = ('original_mp4', 'original_png')
PREP_SHA = '13652f32b4bebba0e354804eadf49d0d1ab67bc4c4c24a524b74a243f671a12e'
REPLAY_SHA = '40003eb5e6d30212b0e21d7916ded7ddc6183463f358394a9d33e8ba96709f7d'
C_AUDIT_SHA = 'efa58040e4ac610ad4973c1c5720287ea2bb502106040d56c2644a54e1a38e2d'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class BoundReader:
    def __init__(self, expected=None): self.bindings = dict(expected or {})
    def bind(self, path, expected=None):
        path = Path(path); digest = sha(path)
        if expected is not None and digest != expected: raise ValueError('Bound source differs: ' + str(path))
        if str(path) in self.bindings and self.bindings[str(path)] != digest: raise ValueError('Source mutated: ' + str(path))
        self.bindings[str(path)] = digest
        return path
    def read(self, path, expected=None): return json.loads(self.bind(path, expected).read_text())
    def rows(self, path, expected=None): return [json.loads(s) for s in self.bind(path, expected).read_text().splitlines()]
    def verify(self):
        for path, digest in list(self.bindings.items()): self.bind(path, digest)


def read_status(path):
    for delay in (0., .05, .1, .2, .4, .8):
        if delay: time.sleep(delay)
        try: return json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if delay == .8: raise


def verify_dependency(folder):
    state = read_status(folder / 'status.json')
    if state['status'] == 'failed': raise RuntimeError('Preceding diagnostic failed')
    if state['status'] == 'completed': return state
    path = Path('/proc') / str(state['pid']) / 'cmdline'
    command = path.read_bytes().decode().replace('\0', ' ') if path.exists() else ''
    if str(folder / 'code/report_native_balanced_scope_control.py') not in command:
        state = read_status(folder / 'status.json')
        if state['status'] != 'completed': raise RuntimeError('Missing actual dependency handle')
    return state


def prepare(a):
    from r3d_encoding_source_input import observation_descriptor, require_same_observation, validate_descriptor
    out = a.output; out.mkdir(parents=True, exist_ok=False); reader = BoundReader()
    if shutil.disk_usage(out).free < 40 * 1024**3: raise RuntimeError('Keep40GiB reserve')
    pr = reader.read(a.prepared / 'report.json', PREP_SHA)
    rp = reader.read(a.observation_check, REPLAY_SHA)
    sources = reader.read(a.prepared / 'sources.json', pr['sources_sha256'])
    if not pr['completed'] or not rp['completed'] or len(sources) != 8: raise ValueError('Complete source preparation required')
    identities = {(i['sample_id'], i['mode']): i for i in rp['identities']}
    if len(identities) != 16: raise ValueError('Complete dual-mode replay required')
    for source in sources:
        if source['split'] != 'train' or source['frame_count'] != 180: raise ValueError('Fixed Train4 sources only')
        validate_descriptor(observation_descriptor(source))
        for mode in MODES:
            require_same_observation(identities[source['sample_id'], mode], identities[source['sample_id'], mode])
    audit_path = C_TRIAL / 'formal_audit/audit.json'
    audit = reader.read(audit_path, C_AUDIT_SHA); old = audit['source_hashes']
    if audit['passed'] is not True: raise ValueError('Original C independent audit required')
    cp = reader.read(C_TRIAL / 'formal/protocol.json', old[str(C_TRIAL / 'formal/protocol.json')])
    ep = reader.read(EVIDENCE / 'protocol.json', old[str(EVIDENCE / 'protocol.json')])
    hp = reader.read(TEACHER / 'protocol.json', ep['source_hashes'][str(TEACHER / 'protocol.json')])
    checkpoint = C_TRIAL / 'formal/C_zero_motion/checkpoint-0003/training_state.pt'
    reader.bind(checkpoint, '5cbbd8c4cbe5efa30a29115a3b27ed2b594e8607613fe9379f14374cc3229a98')
    option_paths = [Path(p) for p in ep['source_hashes'] if p.endswith('/c_protocol.json')]
    if len(option_paths) != 1: raise ValueError('Unique historical command required')
    options = reader.read(option_paths[0], ep['source_hashes'][str(option_paths[0])])['baseline_command'][1:]
    selection_path = Path(hp['backbone_selection'])
    selection = reader.read(selection_path, ep['source_hashes'][str(selection_path)])
    reader.bind(selection_path.parent / selection['selected_checkpoint'], selection['selected_checkpoint_sha256'])
    head = TEACHER / 'H_with_hard_negatives/checkpoint-0003/temporal_head.pt'
    reader.bind(head, ep['source_hashes'][str(head)])
    ids = {s['sample_id'] for s in sources}
    path = EVIDENCE / ep['cohort_manifests']['train_paired']
    records = {r['sample_id']: r for r in reader.rows(path, ep['output_sha256'][path.name]) if r['sample_id'] in ids}
    folder = checkpoint.parent / 'evaluation/train_paired'
    native = {r['sample_id']: r for r in reader.rows(folder / 'predictions.jsonl', old[str(folder / 'predictions.jsonl')]) if r['sample_id'] in ids}
    masks = {r['sample_id']: r for r in reader.read(folder / 'mask_inventory.json', old[str(folder / 'mask_inventory.json')]) if r['sample_id'] in ids}
    path = EVIDENCE / 'train_paired_predictions.jsonl'
    upstream = {r['sample_id']: r for r in reader.rows(path, ep['output_sha256'][path.name]) if r['sample_id'] in ids}
    if any(set(rows) != ids for rows in (records, native, masks, upstream)): raise ValueError('Incomplete original baselines')
    for source in sources:
        r = records[source['sample_id']]
        if r['split'] != 'train' or r['video_path'] != source['mp4_path'] or r['video_sha256'] != source['mp4_sha256']:
            raise ValueError('Original Train source identity differs')
        reader.bind(masks[r['sample_id']]['path'], masks[r['sample_id']]['sha256'])
    gap = reader.read(GAP_AUDIT / 'report.json', 'd99c8a9e7de9972e560a13d62478d31c03d76e2bdf0fea3f1c7933cd773ee9e1')
    gap_cases = reader.read(GAP_AUDIT / 'cases.json', gap['output_sha256']['cases.json'])
    if {r['sample_id'] for r in gap_cases if r['short_gaps']} != {sid for sid in ids if records[sid]['role'] == 'bug'}:
        raise ValueError('Use all four previously selected short-gap sources')
    here = Path(__file__).resolve().parent
    common_files = (Path(__file__).name, 'r3d_encoding_source_input.py', 'r3d_native_video_store.py', 'report_native_tolerant_events.py')
    native_files = ('r3d_native_motion_cues.py', 'r3d_native_restoration_aux.py', 'r3d_native_pixel_refiner.py',
                    'r3d_native_pixel_data.py', 'r3d_native_zero_motion_inference.py')
    code_hashes = {}
    for stage in ('manager', 'upstream', 'native'):
        code = out / (stage + '_code'); code.mkdir()
        if stage != 'manager':
            for file in (TEACHER / 'code').glob('*.py'):
                reader.bind(file, hp['source_sha256'].get(file.name)); shutil.copy2(file, code / file.name)
        names = common_files + (('r3d_label_free_upstream.py',) if stage == 'upstream' else native_files if stage == 'native' else ())
        for name in names:
            file = here / name
            if name in cp['code_sha256'] and name in native_files: reader.bind(file, cp['code_sha256'][name])
            if name in rp['code_sha256']: reader.bind(file, rp['code_sha256'][name])
            shutil.copy2(file, code / name)
        code_hashes.update({str(file): sha(file) for file in code.glob('*.py')})
    dependency = reader.bind(a.after / 'code/report_native_balanced_scope_control.py',
        '58a4619e050cf444dcc11fdbd43cd6fd68134202163a6f43117427d51f0a85c5')
    verify_dependency(a.after)
    save(out / 'sources.json', [observation_descriptor(s) for s in sources])
    save(out / 'expected_identities.json', rp['identities'])
    save(out / 'baselines.json', {sid:dict(upstream=upstream[sid]['frame_probabilities'],
        native=native[sid]['frame_probabilities'], mask_path=masks[sid]['path'], mask_sha256=masks[sid]['sha256']) for sid in sorted(ids)})
    save(out / 'evaluation_records.json', list(records.values()))
    save(out / 'original_frame_targets.json', {sid:native[sid]['frame_targets'] for sid in sorted(ids)})
    save(out / 'gap_cases.json', [r for r in gap_cases if r['sample_id'] in ids])
    for path in out.glob('*.json'): reader.bind(path)
    protocol = dict(protocol='fixed_C_consistent_encoding_train4_v1', modes=list(MODES),
        prepared=str(a.prepared), after_diagnostics=str(a.after), original_C_checkpoint=str(checkpoint),
        upstream_head=str(head), backbone_selection=str(selection_path), historical_options=options,
        sources=8, pairs=4, original_frames=1440, native_inferences=16, upstream_inferences=16,
        threshold=.5, tolerance_frames=3, core_frames=8, halo_frames=4, automatic_promotion=False,
        upstream_precision='Historical BF16; matmulTF32=false,cudnnTF32=true,deterministic=false,benchmark=false',
        native_precision='FP32; noAMP/noTF32; deterministic', model_updates=0,
        all_original_frames=True, original_fps=True, native_resolution=True,
        paired_clean_enters_bug_forward=False, gt_enters_forward=False, development_used=False,
        test_used=False, real_gt_used=False, native_mask_replay='exact packed binary and PTS',
        score_replay_atol=1e-6, score_replay_rtol=1e-6, code_sha256=code_hashes,
        source_sha256=reader.bindings, gpu_free_required_gib=20, disk_reserve_gib=40,
        limitations=['Fixed trained model sensitivity, not retraining or broad efficacy.',
            'Four Train Cache pairs selected by existing short gaps, not random or unseen examples.',
            'PNG clock comes from matched original MP4; no intrinsic PNG time metadata.',
            'Changing encoding does not by itself establish the cause of training failure or fix real deployment.'])
    reader.verify(); save(out / 'protocol.json', protocol)
    save(out / 'preflight.json', dict(passed=True, model_inference=False, training_updates=0,
        source_videos=8, protocol_sha256=sha(out / 'protocol.json'), test_used=False, real_gt_used=False))
    return protocol


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str((array.dtype, tuple(array.shape))).encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def require_baseline_scores(actual, expected):
    import numpy as np
    actual=np.asarray(actual);expected=np.asarray(expected)
    if (actual.ndim!=1 or actual.shape!=expected.shape or not len(actual)
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or (actual<0).any() or (actual>1).any() or (expected<0).any() or (expected>1).any()
            or not np.allclose(actual,expected,atol=1e-6,rtol=1e-6)):
        raise ValueError('Original baseline scores differ; fixed tolerance must not be widened')
    return float(np.abs(actual-expected).max())


def run_model_stage(out, protocol, stage, update):
    import numpy as np
    import torch
    from r3d_encoding_source_input import load_encoding_observations, require_same_observation
    reader = BoundReader(protocol['source_sha256']); reader.verify()
    sources = reader.read(out / 'sources.json'); baselines = reader.read(out / 'baselines.json')
    identities = {(r['sample_id'], r['mode']):r for r in reader.read(out / 'expected_identities.json')}
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        raise RuntimeError('Need20GiB free; never evict another process')
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.65)
    folder = out / stage; folder.mkdir(exist_ok=False); inventory = []
    if stage == 'upstream':
        import train_r3d18_flicker_detector as common
        import train_r3d18_exact_frame_localizer as exact
        from r3d_label_free_upstream import predict_upstream_frames
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.deterministic=False; torch.backends.cudnn.benchmark=False
        argv = sys.argv
        try: sys.argv=protocol['historical_options']; options=exact.parse_args()
        finally: sys.argv=argv
        if str(options.backbone_selection)!=protocol['backbone_selection'] or options.difference_context!='window_zero':
            raise ValueError('Historical upstream options differ')
        backbone, _, _ = exact.load_frozen_backbone(options.backbone_selection)
        head=exact.create_temporal_head(options.head_channels,options.window_frames,options.feature_mode,options.dense_difference_branch)
        head.load_state_dict(torch.load(protocol['upstream_head'],map_location='cuda:0',weights_only=True)); head.eval().requires_grad_(False)
        models=[backbone,head]
    else:
        from r3d_native_motion_cues import NativeMotionCueRefiner
        from r3d_native_zero_motion_inference import predict_zero_motion_chunks,collect_native_chunks
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
        loaded=torch.load(protocol['original_C_checkpoint'],map_location='cpu',weights_only=False)
        if loaded['arm']!='C_zero_motion' or loaded['step']!=300: raise ValueError('Original C300 required')
        native=NativeMotionCueRefiner(False).cuda().eval().requires_grad_(False); native.load_state_dict(loaded['model'],strict=True)
        del loaded; models=[native]
        up=reader.read(out/'upstream/report.json')
        if not up['completed'] or len(up['inventory'])!=16: raise ValueError('Complete same-source upstream required')
        upstream={(r['identity']['sample_id'],r['identity']['mode']):r for r in up['inventory']}
    before=[model_hash(m) for m in models]
    for mode in MODES:
        for index, source in enumerate(sources, 1):
            sid=source['sample_id']; update(status='predicting_'+stage,mode=mode,sample_id=sid,source_index=index,source_total=8)
            if shutil.disk_usage(out).free < 40*1024**3: raise RuntimeError('Keep40GiB reserve')
            rgb,times,identity=load_encoding_observations(source,mode)
            require_same_observation(identity,identities[sid,mode])
            token=hashlib.sha256(sid.encode()).hexdigest()[:16]+'_'+mode
            if stage=='upstream':
                scores,coverage=predict_upstream_frames(common,exact,backbone,head,rgb,
                    window_frames=options.window_frames,window_stride=options.window_stride,
                    feature_mode=options.feature_mode,difference_context=options.difference_context)
                if mode=='original_mp4': require_baseline_scores(scores,baselines[sid]['upstream'])
                result=dict(identity=identity,frame_probabilities=scores.tolist(),coverage=coverage.tolist(),source_time_s=times.tolist())
                path=folder/(token+'.json');save(path,result)
            else:
                entry=upstream[sid,mode]; context=reader.read(entry['path'],entry['sha256'])
                require_same_observation(identity,context['identity'])
                if not np.array_equal(times,context['source_time_s']): raise ValueError('Upstream clock differs')
                packed,scores=collect_native_chunks(predict_zero_motion_chunks(native,rgb,context['frame_probabilities'],source['fps'],times),tuple(rgb.shape[:3]),times)
                if mode=='original_mp4':
                    ref=baselines[sid]; reader.bind(ref['mask_path'],ref['mask_sha256'])
                    with np.load(ref['mask_path'],allow_pickle=False) as original:
                        if (not np.array_equal(packed,original['packed_masks']) or not np.array_equal(times,original['source_time_s'])
                                or not np.array_equal(original['native_shape'],rgb.shape[:3]) or float(original['pixel_threshold'])!=.5):
                            raise ValueError('Original native C replay differs; do not attribute changes to encoding')
                    require_baseline_scores(scores,ref['native'])
                path=folder/(token+'.npz')
                np.savez_compressed(path,packed_masks=packed,frame_probabilities=scores,source_time_s=times,
                                    native_shape=np.asarray(rgb.shape[:3]),pixel_threshold=np.asarray(.5))
            if hashlib.sha256(memoryview(rgb)).hexdigest()!=identity['rgb_sha256']: raise ValueError('Inference mutated source RGB')
            inventory.append(dict(identity=identity,path=str(path),sha256=sha(path)))
            save(folder/'inventory.json',inventory); del rgb,times; torch.cuda.empty_cache()
    if before!=[model_hash(m) for m in models]: raise ValueError('Frozen model changed')
    reader.verify()
    save(folder/'report.json',dict(completed=True,stage=stage,inventory=inventory,model_state_before=before,
        model_parameters_unchanged=True,source_videos=8,full_source_inferences=16,original_frame_uses=2880,
        original_mp4_replays=8,gt_mask_files_opened=0,labels_in_forward=False,paired_clean_in_forward=False,
        training_updates=0,test_used=False,real_gt_used=False,source_sha256=reader.bindings))


def score_saved_outputs(out, protocol):
    import numpy as np
    from PIL import Image
    from report_native_tolerant_events import evaluate
    reader=BoundReader(protocol['source_sha256']); reader.verify()
    records=reader.read(out/'evaluation_records.json'); targets=reader.read(out/'original_frame_targets.json')
    gap_cases={r['sample_id']:r['short_gaps'] for r in reader.read(out/'gap_cases.json')}
    native=reader.read(out/'native/report.json')
    if not native['completed'] or len(native['inventory'])!=16: raise ValueError('Complete native inference required before GT scoring')
    items={(r['identity']['sample_id'],r['identity']['mode']):r for r in native['inventory']}
    summaries={}; output_frames={}
    for mode in MODES:
        predictions=[]; frame_totals=dict(tp=0,fp=0,fn=0,tn=0); gap_total=gap_fp=gap_pixels=0; case_ious=[]; rows=[]
        for record in records:
            sid=record['sample_id']; entry=items[sid,mode]; reader.bind(entry['path'],entry['sha256'])
            with np.load(entry['path'],allow_pickle=False) as result:
                n,h,w=result['native_shape'].tolist(); scores=result['frame_probabilities'].copy()
                masks=np.unpackbits(result['packed_masks'],axis=1,bitorder='little',count=h*w).reshape(n,h,w).astype(bool)
            truth={}
            if record['role']=='bug':
                manifest=reader.read(record['effect_mask_manifest'],record['effect_mask_manifest_sha256'])
                truth={v['frame_index']:v for v in manifest['effect_masks']}
            y=np.asarray([i in truth for i in range(n)],bool); p=masks.any((1,2))
            if not np.array_equal(y,targets[sid]) or not np.array_equal(p,scores>=.5): raise ValueError('GT/frame decisions differ')
            counts=[]; ious=[]
            for t in range(n):
                target=np.zeros((h,w),bool)
                if t in truth:
                    item=truth[t];reader.bind(item['path'],item['sha256'])
                    with Image.open(item['path']) as image: target=np.asarray(image.convert('L'))>0
                    if target.shape!=(h,w) or int(target.sum())!=item['changed_pixel_count']: raise ValueError('GT pixel count differs')
                pred=masks[t]; tp=int((pred&target).sum()); pp=int(pred.sum()); gt=int(target.sum())
                counts.append(dict(frame_index=t,gt_pixels=gt,pred_pixels=pp,tp=tp,fp=pp-tp,fn=gt-tp))
                if gt: ious.append(tp/(gt+pp-tp))
            if ious: case_ious.append(float(np.mean(ious)))
            for key,value in dict(tp=(y&p).sum(),fp=(~y&p).sum(),fn=(y&~p).sum(),tn=(~y&~p).sum()).items(): frame_totals[key]+=int(value)
            gap_indices=[t for a,b in gap_cases.get(sid,[]) for t in range(a,b+1)]
            if any(y[t] for t in gap_indices): raise ValueError('Frozen gap became positive')
            gap_total+=len(gap_indices);gap_fp+=sum(bool(p[t]) for t in gap_indices);gap_pixels+=sum(counts[t]['pred_pixels'] for t in gap_indices)
            predictions.append(dict(sample_id=sid,frame_count=n,fps=record['fps'],frame_targets=y.astype(int).tolist(),frame_probabilities=scores.tolist(),all_source_frames_covered=True))
            rows.append(dict(sample_id=sid,role=record['role'],camera_condition=record['camera_condition'],frames=counts))
        if gap_total!=152: raise ValueError('Full frozen gap denominator required')
        t=frame_totals;denom=2*t['tp']+t['fp']+t['fn']
        summaries[mode]=dict(frame_counts=t,frame_f1=2*t['tp']/denom if denom else None,
            frame_recall=t['tp']/(t['tp']+t['fn']) if t['tp']+t['fn'] else None,
            endpoint_pm3=evaluate(predictions,3),mean_case_positive_frame_iou=float(np.mean(case_ious)),
            normal_gap_frames=gap_total,normal_gap_false_positive_frames=gap_fp,normal_gap_false_positive_pixels=gap_pixels)
        save(out/(mode+'_pixel_frames.json'),rows);save(out/(mode+'_predictions.json'),predictions)
        output_frames[mode]=sha(out/(mode+'_pixel_frames.json'))
    reader.verify()
    return dict(completed=True,fixed_C_encoding_sensitivity=True,results=summaries,
        source_videos=8,pairs=4,upstream_inferences=16,native_inferences=16,training_updates=0,
        automatic_promotion=False,threshold=.5,tolerance_frames=3,development_used=False,test_used=False,real_gt_used=False,
        source_sha256=reader.bindings,pixel_output_sha256=output_frames,
        limitations=protocol['limitations'],independent_output_audit_pending=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','observation-check','after','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--launch',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--stage',choices=('upstream','native'))
    a=parser.parse_args();out=a.output
    if not a.worker:
        prepare(a)
        if a.preflight_only:
            print(json.dumps(dict(preflight_passed=True,output=str(out),model_inference=False)));return
        command=[sys.executable,str(out/'manager_code'/Path(__file__).name),*[s for s in sys.argv[1:] if s!='--launch'],'--worker']
        if a.launch:
            with (out/'worker.log').open('x') as log: child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(out/'launch.json',dict(pid=child.pid,command=command));print(json.dumps(dict(pid=child.pid,output=str(out))));return
        os.execv(sys.executable,command)
    status_name=(a.stage+'_status.json') if a.stage else 'status.json'
    state=dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat())
    def update(**kwargs):
        state.update(kwargs);state['updated_utc']=datetime.now(timezone.utc).isoformat();save(out/status_name,state)
    try:
        protocol=json.loads((out/'protocol.json').read_text())
        if sha(out/'protocol.json')!=json.loads((out/'preflight.json').read_text())['protocol_sha256']: raise ValueError('Frozen protocol changed')
        for path,digest in protocol['code_sha256'].items():
            if sha(path)!=digest: raise ValueError('Snapshot implementation changed')
        if a.stage:
            update(status='preflight_model_stage');run_model_stage(out,protocol,a.stage,update)
        else:
            while True:
                dependency=verify_dependency(a.after)
                if dependency['status']=='completed': break
                update(status='waiting_for_verified_B_diagnostics',dependency_pid=dependency['pid'],dependency_stage=dependency['status']);time.sleep(15)
            completed=json.loads((a.after/'report.json').read_text())
            if not completed['completed'] or not completed['balanced_scope_control']: raise ValueError('Required full B diagnostics absent')
            for stage in ('upstream','native'):
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                command=[sys.executable,str(out/(stage+'_code')/Path(__file__).name),
                    '--prepared',str(a.prepared),'--observation-check',str(a.observation_check),'--after',str(a.after),
                    '--output',str(out),'--worker','--stage',stage]
                with (out/(stage+'.log')).open('x') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                    update(status='running_'+stage,child_pid=child.pid)
                    if child.wait()!=0: raise RuntimeError('Model stage failed; preserve output: '+stage)
            update(status='scoring_complete_outputs');save(out/'report.json',score_saved_outputs(out,protocol))
        update(status='completed')
    except BaseException as error:
        update(status='failed',error=repr(error));raise


if __name__=='__main__': main()
