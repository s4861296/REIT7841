
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from run_flicker_multimodel_pilot import key_for, load_inputs, save, sha, utc
from flicker_multimodel_streaming import create_observed_store, complete_dense_prediction
from flicker_multimodel_observed_features import assemble_observed_features, pooled_motion_cues

BASE = Path('/mnt/d/VRFlickerLab')
TEACHER = BASE / 'model_training/r3d18_flicker_v1/factorial100_normal_consistency_20260921_v1/without_consistency'
EVIDENCE = BASE / 'model_evaluation/r3d18_flicker_v1/motion_fusion_native_evidence_20260921_v1'
FLOW = BASE / 'model_evaluation/r3d18_flicker_v1/native_motion_full_cache_recovery_20260925_v2/development200'
DEFAULT_MANIFEST = BASE / 'model_evaluation/qwen25_vl_earlystop_2240_gt_20260904/complete_video_manifest_scored.jsonl'


def prepare(parent, following, manifest, out):
    if out.exists(): raise FileExistsError('Use fresh inference output')
    out.parent.mkdir(parents=True, exist_ok=True)
    p, inputs = load_inputs(parent)
    if p['scope'] != 'full_train300_dev200': raise ValueError('Require full parent')
    if shutil.disk_usage(out.parent).free < 80 * 1024**3: raise OSError('Keep80GiB before native inference')
    ep = json.loads((EVIDENCE / 'protocol.json').read_text())
    hp = json.loads((TEACHER / 'protocol.json').read_text())
    source = [json.loads(line) for line in manifest.read_text().splitlines()]
    
    if len(source) != 21 or any(not r['scored'] or r['gt_label'] not in ('positive', 'negative') for r in source):
        raise ValueError('Expected full previously evaluated21-source cohort')
    observed = []
    for r in source:
        observed.append({'sample_id': r['clip_id'], 'video_path': r['video_path'], 'video_sha256': sha(r['video_path']),
            'approved_seconds': [r['window_start_s'], r['window_end_s']] if r['human_restricted_range'] else None})
    
    bug = min((i for i in inputs['train'] if i['record']['role'] == 'bug'), key=lambda i: i['record']['sample_id'])
    pair = [i for i in inputs['train'] if i['record']['pair_id'] == bug['record']['pair_id']]
    hn = min((i for i in inputs['train'] if i['record']['role'] == 'hard_negative'), key=lambda i: i['record']['sample_id'])
    replay = []
    for i in sorted(pair, key=lambda i: i['record']['role']) + [hn]:
        obs = i['observed']['observation']; sid = obs['sample_id']
        replay.append({'sample_id': sid, 'video_path': obs['video_path'], 'video_sha256': obs['video_sha256'],
            'approved_seconds': None, 'expected_rgb_sha256': obs['rgb_sha256'],
            'expected_context': i['observed']['context']['frame_probabilities'], 'expected_reference': i['observed']['reference'],
            'expected_feature': str(parent / 'features' / (key_for(sid) + '.npy')),
            'expected_prediction': str(parent / 'predictions' / (key_for(sid) + '.json'))})
    selection_path = Path(hp['backbone_selection']); selection = json.loads(selection_path.read_text())
    head = TEACHER / 'H_with_hard_negatives/checkpoint-0003/temporal_head.pt'
    dense_parent = json.loads((BASE / 'model_training/r3d18_flicker_v1/dense_early_native_train100_formal_20261005_v2/protocol.json').read_text())
    flow = json.loads((FLOW / 'protocol.json').read_text())
    weight = next(Path(f) for f in flow['input_sha256'] if 'raft_large_C_T_SKHT_V2-' in f)
    models = {'backbone_selection': str(selection_path), 'backbone': str(selection_path.parent / selection['selected_checkpoint']),
        'head': str(head), 'native_C': dense_parent['original_C_checkpoint'], 'raft': str(weight), 'dino': p['weights']}
    bindings = {str(selection_path): sha(selection_path), models['backbone']: selection['selected_checkpoint_sha256'],
        str(head): ep['source_hashes'][str(head)], models['native_C']: dense_parent['original_C_checkpoint_sha256'],
        str(weight): flow['producer']['weights_sha256']}
    for name, entry in p['weights_provenance']['files'].items(): bindings[str(Path(p['weights']) / name)] = entry['sha256']
    for path, digest in bindings.items():
        if sha(path) != digest: raise ValueError('Frozen model differs: ' + path)
    out.mkdir(); (out / 'code').mkdir()
    here = Path(__file__).resolve().parent
    from run_r3d_audited_native_transfer import python_closure
    files = python_closure(here, Path(__file__).name)
    for name in ('build_native_motion_train_cache.py', 'train_r3d18_exact_frame_localizer.py',
                 'r3d_label_free_upstream.py', 'r3d_native_zero_motion_inference.py', 'flicker_multimodel_fast_state.py'):
        files.update(python_closure(here, name))
    for name, path in files.items():
        if name in p['code_sha256'] and sha(path) != p['code_sha256'][name]: raise ValueError('Parent code differs: ' + name)
        shutil.copy2(path, out / 'code' / name)
    for name in ('train_r3d18_flicker_detector.py', 'train_r3d18_exact_frame_localizer.py'):
        if sha(out / 'code' / name) != sha(TEACHER / 'code' / name): raise ValueError('Historical H implementation changed')
    for name, digest in flow['implementation']['code'].items():
        if name in files and sha(files[name]) != digest: raise ValueError('RAFT producer implementation changed: ' + name)
    save(out / 'observations.json', observed); save(out / 'replay_observations.json', replay)
    tests = ['test_flicker_multimodel_streaming.py', 'test_flicker_multimodel_observed_features.py']
    for name in tests: shutil.copy2(here.parent / 'tests' / name, out / name)
    protocol = {'version': 'frozen_multimodel_observation_chain_v1', 'prepared_at': utc(), 'parent': str(parent),
        'parent_checkpoint_sha256': sha(parent / 'latest_checkpoint.pt'), 'parent_protocol_sha256': sha(parent / 'protocol.json'),
        'normalization_sha256': sha(parent / 'normalization.npz'),
        'following': str(following), 'following_protocol_sha256': sha(following / 'protocol.json'),
        'candidate_policy': 'Parent DINO and ALL4 final factorial arms; no real-based selection or threshold fit',
        'models': models, 'model_sha256': bindings, 'source_manifest': str(manifest), 'source_manifest_sha256': sha(manifest),
        'historical_real_set_reused': True, 'new_holdout': False, 'classification_gt_used_only_after_prediction': True,
        'training_updates': 0, 'threshold': .5, 'all_original_frames': True, 'source_resize': False,
        'fps_resampled': False, 'native_source_pts_preserved': True, 'temporal_head_core': 32, 'temporal_head_halo': 6,
        'feature_core': 32, 'feature_halo': 8, 'upstream_window': 32, 'upstream_stride': 8,
        'upstream_difference_context': 'window_zero', 'upstream_feature_mode': 'appearance_delta_mean_max',
        'upstream_dense_difference_branch': 'shallow_native', 'upstream_head_channels': 64,
        'raft_producer': flow['producer'], 'flow_storage': 'Same native flow; mean/max pooling identical to Train; accepts monotonic original VFR PTS',
        'replay_gate': '3 complete Train sources: RGB exact; H/C scores atol1e-6; C masks exact; parent final probabilities atol1e-5 and decisions exact',
        'real_localization_quantitative': False, 'real_localization_reason': 'Historical labels are not independently verified exact per-frame/pixel GT',
        'reviews': 'Parent DINO only, original source PTS intervals; old C mask gated by new state, no GT overlay',
        'automatic_promotion': False, 'test_used': False, 'real_gt_used': 'post-inference historical binary diagnostic only',
        'code_sha256': {name: sha(out / 'code' / name) for name in files},
        'input_sha256': {name: sha(out / name) for name in ('observations.json', 'replay_observations.json')},
        'test_sha256': {name: sha(out / name) for name in tests},
        'limitations': ['Repeated historical real cohort, NOT new holdout.', 'VFR accepted without resampling; trained models are not proven frame-rate invariant.',
            '21 sources include5 negatives only; not sufficient broad real-world validation.', 'Spatial output reuses old C boundaries.',
            'Real Root cause and exact time/pixel accuracy are not inferred from binary labels.']}
    save(out / 'protocol.json', protocol)
    env = dict(os.environ, PYTHONPATH=str(out / 'code'), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    checked = subprocess.run([sys.executable, '-m', 'unittest', *(n[:-3] for n in tests), '-v'], cwd=out,
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    save(out / 'tests.json', {'returncode': checked.returncode, 'output': checked.stdout})
    if checked.returncode: raise RuntimeError(checked.stdout)
    print(json.dumps({'prepared': str(out), 'real_sources': len(observed), 'replay_sources': len(replay)}), flush=True)


def upstream(folder, p, rgb, update):
    import torch
    path = folder / 'upstream.json'
    if path.exists(): return json.loads(path.read_text())
    import train_r3d18_flicker_detector as common
    import train_r3d18_exact_frame_localizer as exact
    from r3d_label_free_upstream import predict_upstream_frames
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.deterministic = False
    backbone, _, _ = exact.load_frozen_backbone(Path(p['models']['backbone_selection']))
    head = exact.create_temporal_head(p['upstream_head_channels'], p['upstream_window'],
        p['upstream_feature_mode'], p['upstream_dense_difference_branch'])
    head.load_state_dict(torch.load(p['models']['head'], map_location='cuda:0', weights_only=True)); head.eval().requires_grad_(False)
    update(status='predicting_historical_R3D_H', original_frames=len(rgb))
    scores, coverage = predict_upstream_frames(common, exact, backbone, head, rgb,
        window_frames=p['upstream_window'], window_stride=p['upstream_stride'],
        feature_mode=p['upstream_feature_mode'], difference_context=p['upstream_difference_context'])
    result = {'frame_probabilities': scores.tolist(), 'coverage': coverage.tolist(), 'labels_used': False}
    save(path, result); del head, backbone; torch.cuda.empty_cache()
    return result


def native_mask(folder, p, rgb, observation, scores, update):
    import torch
    from r3d_native_motion_cues import NativeMotionCueRefiner
    from r3d_native_zero_motion_inference import predict_zero_motion_chunks, collect_native_chunks
    path = folder / 'native_C.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as z: return z['packed_masks'].copy(), z['frame_probabilities'].copy()
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    m = NativeMotionCueRefiner(False).cuda().eval().requires_grad_(False)
    c = torch.load(p['models']['native_C'], map_location='cpu', weights_only=False)
    if c['arm'] != 'C_zero_motion' or c['step'] != 300: raise ValueError('Wrong original C checkpoint')
    m.load_state_dict(c['model']); del c
    update(status='predicting_native_C', original_frames=len(rgb))
    times = np.asarray(observation['source_time_s'], np.float64)
    def chunks():
        for row in predict_zero_motion_chunks(m, rgb, scores, observation['fps'], times):
            update(status='predicting_native_C', frame=row['end_frame_exclusive'], original_frames=len(rgb)); yield row
    packed, maxima = collect_native_chunks(chunks(), tuple(rgb.shape[:3]), times)
    with path.open('xb') as f: np.savez_compressed(f, packed_masks=packed, frame_probabilities=maxima, source_time_s=times)
    del m; torch.cuda.empty_cache()
    return packed, maxima


def motion(folder, p, rgb, update):
    import torch
    from build_native_motion_train_cache import make_raft_inference
    path = folder / 'motion_pooled.npy'; meta = folder / 'motion_pooled.json'
    if meta.exists():
        if sha(path) != json.loads(meta.read_text())['sha256']: raise ValueError('Motion cache changed')
        return np.load(path, mmap_mode='r')
    if path.exists(): raise FileExistsError('Incomplete motion evidence retained')
    n, h, w, _ = rgb.shape
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    infer = make_raft_inference(p['models']['raft'])
    stored = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n, 5, (h + 13)//14, (w + 13)//14))
    stored[0] = pooled_motion_cues(np.zeros((h,w), np.float32), np.zeros((h,w), bool), np.zeros((h,w), bool), False)
    for t in range(1, n):
        aligned, support, consistent = infer(rgb[t-1], rgb[t])
        stored[t] = pooled_motion_cues(aligned, support, consistent, True)
        if t % 16 == 0 or t + 1 == n:
            stored.flush(); update(status='native_RAFT', frame=t+1, original_frames=n)
    stored.flush()
    save(meta, {'sha256': sha(path), 'frames': n, 'directed_flow_passes': 2*(n-1), 'native_flow_before_pooling': True,
                'source_resize': False, 'gt_used': False, 'producer': p['raft_producer']})
    del infer; torch.cuda.empty_cache()
    return stored


def features(folder, p, rgb, scores, packed_masks, pooled, update):
    import torch
    from transformers import Dinov2Model
    from flicker_multimodel_core import dino_embeddings
    path = folder / 'features.npy'; meta = folder / 'features.json'
    if meta.exists():
        if sha(path) != json.loads(meta.read_text())['sha256']: raise ValueError('Feature cache changed')
        return np.load(path, mmap_mode='r')
    if path.exists(): raise FileExistsError('Incomplete features retained')
    torch.use_deterministic_algorithms(False); torch.backends.cudnn.allow_tf32 = False
    m = Dinov2Model.from_pretrained(p['models']['dino'], local_files_only=True, attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    n, h, w, _ = rgb.shape
    output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(n, 51, (h+13)//14, (w+13)//14))
    names = None
    for first in range(0, n, p['feature_core']):
        last = min(n, first + p['feature_core']); left = max(0, first-p['feature_halo']); right = min(n, last+p['feature_halo'])
        embedding = dino_embeddings(m, rgb[left:right])
        masks = np.unpackbits(packed_masks[left:right], axis=1, count=h*w, bitorder='little').reshape(right-left,h,w).astype(bool)
        block, names = assemble_observed_features(rgb[left:right], embedding, scores[left:right], masks, pooled[left:right])
        output[first:last] = block[first-left:last-left]; output.flush()
        update(status='native_DINO_feature_stream', frame=last, original_frames=n)
        del embedding, masks, block
    save(meta, {'sha256': sha(path), 'frames': n, 'feature_names': names, 'all_original_frames': True,
        'native_source_resolution': [h,w], 'feature_halo': p['feature_halo'], 'gt_used': False})
    del m; torch.cuda.empty_cache()
    return output


def predict_candidates(folder, p, observed_features, old_scores):
    import torch
    from flicker_multimodel_core import DenseStateHead
    from flicker_multimodel_fast_state import ARMS, FastResidualStateHead
    dest = folder / 'predictions.json'
    if dest.exists(): return json.loads(dest.read_text())
    parent, follow = Path(p['parent']), Path(p['following'])
    cp = torch.load(parent / 'latest_checkpoint.pt', map_location='cpu', weights_only=False)
    later = torch.load(follow / 'latest_checkpoint.pt', map_location='cpu', weights_only=False)
    with np.load(parent / 'normalization.npz') as z: mean, std = z['mean'].copy(), z['std'].copy()
    definitions = [('parent_DINO', DenseStateHead, cp['models']['motion_context_dino'])]
    definitions += [(a, FastResidualStateHead if c['fast'] else DenseStateHead, later['models'][a]) for a,c in ARMS.items()]
    result = {'frozen_C': {'frame_probabilities': list(map(float, old_scores))}}
    for name, cls, state in definitions:
        m = cls(cp['channels']).cuda().eval(); m.load_state_dict(state)
        x = complete_dense_prediction(m, observed_features, mean, std, core_frames=p['temporal_head_core'])
        result[name] = {k: v.tolist() for k,v in x.items()}
        del m
    save(dest, result); torch.cuda.empty_cache()
    return result


def one_source(out, p, source, stage, update):
    folder = out / stage / key_for(source['sample_id']); folder.mkdir(parents=True, exist_ok=True)
    native = folder / 'source'
    if not native.exists(): obs = create_observed_store(source['video_path'], native, source['video_sha256'], source['approved_seconds'])
    else:
        if json.loads((native / 'status.json').read_text())['status'] != 'completed': raise ValueError('Incomplete native store retained')
        obs = json.loads((native / 'observation.json').read_text())
        if obs['video_sha256'] != source['video_sha256'] or sha(source['video_path']) != obs['video_sha256']: raise ValueError('Original source changed')
    rgb = np.load(native / 'rgb.npy', mmap_mode='r')
    if sha(native / 'rgb.npy') != obs['rgb_npy_sha256']: raise ValueError('Decoded pixels changed')
    up = upstream(folder,p,rgb,update); scores = np.asarray(up['frame_probabilities'],np.float64)
    packed, maxima = native_mask(folder,p,rgb,obs,scores,update)
    pooled = motion(folder,p,rgb,update)
    ft = features(folder,p,rgb,scores,packed,pooled,update)
    prediction = predict_candidates(folder,p,ft,maxima)
    result = {'sample_id': source['sample_id'], 'folder': str(folder), 'observation': obs,
        'prediction_path': str(folder/'predictions.json'), 'prediction_sha256': sha(folder/'predictions.json')}
    if stage == 'replay':
        ref = source['expected_reference']; reference = json.loads(Path(source['expected_prediction']).read_text())
        if obs['rgb_sha256'] != source['expected_rgb_sha256']: raise ValueError('Replay RGB differs')
        np.testing.assert_allclose(scores, source['expected_context'], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(maxima, ref['frame_probabilities'], atol=1e-6, rtol=1e-6)
        if sha(ref['mask_path']) != ref['mask_sha256']: raise ValueError('Replay reference changed')
        with np.load(ref['mask_path'], allow_pickle=False) as z: np.testing.assert_array_equal(packed,z['packed_masks'])
        new = np.asarray(prediction['parent_DINO']['frame_probabilities']); old = np.asarray(reference['predictions']['motion_context_dino']['frame_probabilities'])
        np.testing.assert_allclose(new,old,atol=1e-5,rtol=1e-5); np.testing.assert_array_equal(new>=.5,old>=.5)
        old_features = np.load(source['expected_feature'],mmap_mode='r')
        delta = float(np.max(np.abs(np.asarray(ft,np.float32)-np.asarray(old_features,np.float32))))
        result['replay'] = {'passed': True, 'rgb_exact': True, 'native_C_mask_exact': True,
            'parent_probability_max_abs_error': float(np.abs(new-old).max()), 'derived_feature_max_abs_error': delta,
            'state_decisions_exact': True, 'gt_opened': False}
    save(folder/'source_result.json',result)
    return result


def real_review(result):
    import av
    import cv2
    folder = Path(result['folder']); dest = folder/'parent_DINO_review.mp4'
    if dest.exists(): return
    obs = result['observation']; rgb = np.load(folder/'source/rgb.npy',mmap_mode='r')
    scores = json.loads(Path(result['prediction_path']).read_text())['parent_DINO']['frame_probabilities']
    with np.load(folder/'native_C.npz',allow_pickle=False) as z: packed=z['packed_masks'].copy()
    with av.open(str(dest),'w') as c:
        stream=c.add_stream('libx264',rate=Fraction(obs['fps_fraction']))
        stream.width=obs['width'];stream.height=obs['height'];stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'fast','threads':'2'}
        stream.time_base=Fraction(obs['time_base_fraction']);stream.codec_context.time_base=stream.time_base
        for t in range(len(rgb)):
            image=np.array(rgb[t],copy=True)
            if scores[t]>=.5:
                mask=np.unpackbits(packed[t],count=obs['height']*obs['width'],bitorder='little').reshape(obs['height'],obs['width']).astype(bool)
                image[mask]=(image[mask].astype(np.float32)*.55+np.array([255,0,0])*.45).astype(np.uint8)
            cv2.rectangle(image,(0,0),(obs['width'],44),(0,0,0),-1)
            cv2.putText(image,f'frame={t+obs["source_start_frame"]} source_t={obs["source_time_s"][t]:.3f}s p={scores[t]:.3f}',
                (5,16),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(image,'OLD C region + NEW time; no GT overlay',(5,35),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1,cv2.LINE_AA)
            f=av.VideoFrame.from_ndarray(image,format='rgb24');f.pts=obs['source_pts'][t]-obs['source_pts'][0];f.time_base=Fraction(obs['time_base_fraction'])
            for packet in stream.encode(f): c.mux(packet)
        for packet in stream.encode(): c.mux(packet)
    decoded=[]
    with av.open(str(dest)) as c:
        for f in c.decode(video=0):decoded.append(float(f.pts*f.time_base))
    expected=np.asarray(obs['source_time_s'])-obs['source_time_s'][0]
    if len(decoded)!=len(expected) or not np.allclose(decoded,expected,rtol=0,atol=2e-5):
        raise ValueError('Review did not retain original frame timing; output retained for inspection')


def run(out):
    import torch
    import cv2
    if Path(__file__).resolve().parent!=out.resolve()/'code':raise ValueError('Run frozen snapshot only')
    lock=(out/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=json.loads((out/'protocol.json').read_text());context={}
    def update(**kw):save(out/'status.json',{'pid':os.getpid(),'updated_at':utc(),**context,**kw})
    try:
        for section,prefix in [('code_sha256',out/'code'),('input_sha256',out),('test_sha256',out)]:
            for n,d in p[section].items():
                if sha(prefix/n)!=d:raise ValueError('Frozen file changed: '+n)
        for path,digest in p['model_sha256'].items():
            if sha(path)!=digest:raise ValueError('Frozen model changed')
        if sha(p['source_manifest'])!=p['source_manifest_sha256']:raise ValueError('Historical cohort changed')
        for name,field in [('latest_checkpoint.pt','parent_checkpoint_sha256'),('protocol.json','parent_protocol_sha256'),('normalization.npz','normalization_sha256')]:
            if sha(Path(p['parent'])/name)!=p[field]:raise ValueError('Parent model/config changed')
        following=Path(p['following'])
        if sha(following/'protocol.json')!=p['following_protocol_sha256']:raise ValueError('Factorial protocol changed')
        while True:
            s=json.loads((following/'status.json').read_text())
            if s['status']=='completed':break
            if s['status']=='failed':raise RuntimeError('Factorial failed; do not silently restart')
            cmd=Path('/proc')/str(s['pid'])/'cmdline'
            if not cmd.exists() or str(following).encode() not in cmd.read_bytes():raise RuntimeError('Actual factorial process missing before completion')
            update(status='waiting_actual_factorial',dependency_pid=s['pid'],dependency_stage=s['status']);time.sleep(10)
        final=json.loads((following/'results.json').read_text())
        if sha(following/'latest_checkpoint.pt')!=final['checkpoint_sha256']:raise ValueError('Factorial final weights changed')
        save(out/'frozen_candidate_weights.json',{'parent':p['parent_checkpoint_sha256'],'four_arm_final':final['checkpoint_sha256'],
            'selection':'ALL prespecified arms; not chosen using real outcome'})
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<20*1024**3:raise RuntimeError('Need20GiB free, do not evict another task')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.set_num_threads(2);cv2.setNumThreads(1);torch.cuda.set_per_process_memory_fraction(.65)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
        replays=[]
        for j,source in enumerate(json.loads((out/'replay_observations.json').read_text()),1):
            context={'stage':'replay','source':j,'total_sources':3,'sample_id':source['sample_id']}
            replays.append(one_source(out,p,source,'replay',update));save(out/'replay_results.json',replays)
        if not all(r['replay']['passed'] for r in replays):raise ValueError('Full chain replay must pass before real sources')
        results=[];failures=[]
        for j,source in enumerate(json.loads((out/'observations.json').read_text()),1):
            context={'stage':'historical_real','source':j,'total_sources':21,'sample_id':source['sample_id']}
            try:
                result=one_source(out,p,source,'historical_real',update)
                results.append(result);save(out/'prediction_inventory.json',results)
            except Exception as exc:
                failures.append({'sample_id':source['sample_id'],'error':str(exc),'traceback':traceback.format_exc()})
                save(out/'source_failures.json',failures);torch.cuda.empty_cache();continue
            try:real_review(result)
            except Exception as exc:
                save(Path(result['folder'])/'review_error.json',{'error':str(exc),'predictions_remain_valid':True})
        
        labels={r['clip_id']:r['gt_label']=='positive' for r in [json.loads(l) for l in Path(p['source_manifest']).read_text().splitlines()]}
        scores={};details=[]
        for r in results:
            predictions=json.loads(Path(r['prediction_path']).read_text())
            for arm,row in predictions.items():
                score=max(row['frame_probabilities']); y=labels[r['sample_id']];pred=score>=.5
                counts=scores.setdefault(arm,Counter());counts['tp' if y and pred else 'fp' if pred else 'fn' if y else 'tn']+=1
                details.append({'sample_id':r['sample_id'],'arm':arm,'actual':y,'prediction':pred,'score':score})
        metrics={}
        for a,c in scores.items():
            tp,fp,fn,tn=(c[k] for k in ['tp','fp','fn','tn']);n=tp+fp+fn+tn
            metrics[a]={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'scored_videos':n,
                'accuracy':(tp+tn)/n,'recall':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,
                'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
        save(out/'results.json',{'completed':len(results)==21,'expected_videos':21,'scored_videos':len(results),'failures':failures,
            'metrics':metrics,'details':details,'historical_real_reused':True,'new_holdout':False,
            'threshold_fitted_to_real':False,'real_frame_pixel_localization_scored':False,'automatic_promotion':False})
        import html
        links=''.join('<article><h3>'+html.escape(r['sample_id'])+'</h3><video controls preload="none" src="historical_real/'+key_for(r['sample_id'])+'/parent_DINO_review.mp4"></video></article>' for r in results)
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:system-ui;max-width:1300px;margin:24px auto}video{max-width:100%;max-height:720px}article{margin:30px 0}</style><h1>历史真实视频诊断：Parent DINO</h1><p>不是新holdout。无GT叠加；红色为旧C区域经新时间判断筛选。原帧与原始帧间隔保留。所有候选原始分数在对应predictions.json。</p>'+links)
        context={};update(status='completed' if len(results)==21 else 'completed_with_source_failures',scored_videos=len(results),failures=len(failures),automatic_promotion=False)
    except BaseException as exc:
        update(status='failed',error=str(exc),traceback=traceback.format_exc());raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','pipeline'),required=True)
    parser.add_argument('--parent',type=Path);parser.add_argument('--following',type=Path)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.action=='prepare':prepare(a.parent.resolve(),a.following.resolve(),a.manifest.resolve(),a.output.resolve())
    else:run(a.output.resolve())


if __name__=='__main__':main()
