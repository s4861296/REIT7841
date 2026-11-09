

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.optical_flow import raft_large
import train_r3d18_flicker_detector as common
from probe_native_raft_execution import sha, save, native_frames
from native_dense_flow_evidence import prepare_raft_pair, native_flow_evidence

SOURCE_KEYS = ('sample_id','video_path','video_sha256','frame_count','fps','height','width')


def ordered_train_records(rows, formal):
    if len(rows) != 300 or len({r['sample_id'] for r in rows}) != 300 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Require 300 unique Train videos')
    bugs = [r for r in rows if r['role'] == 'bug']
    hn = [r for r in rows if r['role'] == 'hard_negative']
    roots = Counter(common.root_of(r) for r in bugs)
    categories = Counter(r['hard_negative_category'] for r in hn)
    if len(roots) != 5 or set(roots.values()) != {20} or len(categories) != 5 or set(categories.values()) != {20}:
        raise ValueError('Need 20 pairs per root and 20 HN per category')
    if len(formal['pair_order']) != 100 or len(set(formal['pair_order'])) != 100:
        raise ValueError('Formal pair order invalid')
    selected = []
    for pair in formal['pair_order']:
        local = []
        for role in ('bug','clean_or_fixed'):
            match = [r for r in rows if r['pair_id'] == pair and r['role'] == role]
            if len(match) != 1:
                raise ValueError('Incomplete or duplicate pair')
            local += match
        match = [r for r in hn if r['sample_id'] == formal['hn_pairing'][pair]]
        if len(match) != 1:
            raise ValueError('Incomplete HN assignment')
        for key in ('scene_id','frame_count','fps','height','width'):
            if local[0][key] != local[1][key]:
                raise ValueError('Paired native geometry/timing differs')
        selected += local + match
    if {r['sample_id'] for r in selected} != {r['sample_id'] for r in rows} or len(selected) != 300:
        raise ValueError('Formal mapping duplicates or omits a Train video')
    return selected


def phase_at(targets, t):
    if t == 0:
        return 'first_no_predecessor'
    return ('interior' if targets[t-1] else 'onset') if targets[t] else ('recovery' if targets[t-1] else 'normal')


def residual_statistics(value):
    
    flat = value.float().flatten()
    if flat.numel() < 1 or not torch.isfinite(flat).all() or flat.min() < 0 or flat.max() > 1.000001:
        raise ValueError('Expected finite native residual in [0,1]')
    result = {'mean': flat.mean().item(), 'max': flat.max().item()}
    for fraction, name in ((.001,'top_0p1percent'),(.02,'top_2percent')):
        result[name] = flat.topk(max(1, math.ceil(flat.numel()*fraction)), sorted=False).values.mean().item()
    return result


def region_sums(raw, aligned, mask):
    
    if raw.shape != aligned.shape or mask.shape != raw.shape or mask.dtype != torch.bool:
        raise ValueError('Oracle scoring grid mismatch')
    result = {}
    for name, selected in (('positive', mask), ('negative', ~mask)):
        count = int(selected.sum())
        result[name] = {'pixels': count,
                        'raw_sum': raw[selected].double().sum().item() if count else 0.,
                        'aligned_sum': aligned[selected].double().sum().item() if count else 0.}
    return result


def reduce_group(frames):
    
    valid = [r for r in frames if r['features'] is not None]
    result = {'source_frames': len(frames), 'transition_frames': len(valid),
              'positive_first_frames': sum(r['features'] is None and r['target'] for r in frames)}
    if not valid:
        return result
    for name in ('raw','aligned'):
        result[name] = {k: float(np.mean([r['features'][name][k] for r in valid]))
                        for k in ('mean','max','top_0p1percent','top_2percent')}
    for name in ('positive','negative'):
        pixels = sum(r['oracle_regions'][name]['pixels'] for r in valid)
        result[name+'_region'] = {'pixels': pixels}
        for channel in ('raw','aligned'):
            total = sum(r['oracle_regions'][name][channel+'_sum'] for r in valid)
            result[name+'_region'][channel+'_mean'] = total/pixels if pixels else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','formal-run','execution-probe','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    a = parser.parse_args()
    out = a.output
    if not a.worker:
        out.mkdir(parents=True, exist_ok=False)
        (out/'code').mkdir()
        for name in (Path(__file__).name, 'native_dense_flow_evidence.py', 'probe_native_raft_execution.py',
                     'train_r3d18_flicker_detector.py'):
            shutil.copy2(Path(__file__).with_name(name), out/'code'/name)
        os.execv(sys.executable, [sys.executable, str(out/'code'/Path(__file__).name), *sys.argv[1:], '--worker'])
    state = {'pid': os.getpid(), 'status': 'preparing', 'started_utc': common.utc_now(), 'optimizer_steps': 0}
    def update(**kw):
        state.update(kw)
        save(out/'status.json', state)
    hashes = {}
    def bind(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('Source hash mismatch: '+str(path))
        hashes[str(path)] = actual
        return actual
    def read(path, expected=None):
        bind(path, expected)
        return json.loads(Path(path).read_text())
    update()
    try:
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12*1024**3:
            raise RuntimeError('Need >=12GiB free; do not restart or disturb formal training')
        torch.cuda.set_per_process_memory_fraction(.25)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        ep = read(a.evidence/'protocol.json')
        formal = read(a.formal_run/'protocol.json')
        probe = read(a.execution_probe/'protocol.json')
        audit = read(a.execution_probe/'execution_audit_v2.json')
        if (not audit['passed'] or audit['efficacy_evaluated'] or audit['source_frames'] != 480
                or audit['protocol_sha256'] != hashes[str(a.execution_probe/'protocol.json')]
                or audit['report_sha256'] != bind(a.execution_probe/'report.json')):
            raise ValueError('Need bound full-native execution check; not an efficacy gate')
        probe_inputs = read(a.execution_probe/'input_hashes.json')
        for original in (a.evidence/'protocol.json', a.formal_run/'protocol.json'):
            if probe_inputs.get(str(original)) != hashes[str(original)]:
                raise ValueError('Evidence/formal protocol differs from execution check')
        if probe['code_sha256']['native_dense_flow_evidence.py'] != sha(out/'code/native_dense_flow_evidence.py'):
            raise ValueError('Geometry helper changed since execution check')
        rows = []
        for cohort in ('train_paired','train_hn'):
            file = ep['cohort_manifests'][cohort]
            bind(a.evidence/file, ep['output_sha256'][file])
            shutil.copy2(a.evidence/file, out/file)
            rows += common.read_jsonl(a.evidence/file)
        records = ordered_train_records(rows, formal)
        weight_path = a.execution_probe/'raft_large_C_T_SKHT_V2-ff5fadd5.pth'
        bind(weight_path, probe['weights_sha256'])
        protocol = {
            'purpose': 'Train-only descriptive attenuation/phase/GT-region diagnosis, not efficacy or training',
            'train_videos': 300, 'train_pairs': 100, 'train_hn': 100,
            'source_frames': sum(r['frame_count'] for r in records),
            'directed_flow_passes': sum(2*(r['frame_count']-1) for r in records),
            'weights_sha256': probe['weights_sha256'], 'weights': probe['weights'],
            'iterations': 12, 'dtype': 'float32', 'tf32': False,
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'memory_fraction_cap': .25, 'all_source_frames': True, 'resize': False, 'fps_changed': False,
            'threshold_fitted': False, 'optimizer_steps': 0, 'development_used': False,
            'test_used': False, 'real_gt_used': False, 'active_training_changed': False,
            'features': 'observed adjacent RGB only; no GT/paired-Clean/label in RAFT or warping',
            'oracle_masks': 'AFTER inference reduction only; stored separately from label-blind feature JSONL',
            'quality_mask': 'in-bounds/cycle geometry channels retained, never suppress raw evidence',
            'aggregation': 'frame-weighted group means and pixel-weighted GT regions; separate per-case summaries retained',
            'first_frame': 'decoded and logged with null predecessor features; not automatically Clean',
            'interpretation': 'Low residual is not Clean; recovery change is not necessarily abnormal state; no Bug probability produced',
            'source_order': [r['sample_id'] for r in records],
            'code_sha256': {f.name:sha(f) for f in (out/'code').glob('*.py')},
        }
        save(out/'protocol.json', protocol)
        save(out/'input_hashes.json', hashes)
        model = raft_large(weights=None, progress=False)
        model.load_state_dict(torch.load(weight_path,map_location='cpu',weights_only=True), strict=True)
        model = model.cuda().eval()
        model.requires_grad_(False)
        inventory = []
        groups = defaultdict(list)
        cases = []
        for video_index, record in enumerate(records):
            if shutil.disk_usage(out).free < 10*1024**3:
                raise RuntimeError('Preserve disk reserve; stop without deleting or resizing')
            source = {k: record[k] for k in SOURCE_KEYS}  
            bind(source['video_path'], source['video_sha256'])
            targets = common.frame_targets(record)
            fidelity = 'negative_no_effect'
            masks = {}
            if record['role'] == 'bug':
                manifest = read(Path(record['effect_mask_manifest']), record['effect_mask_manifest_sha256'])
                if any(manifest[k] != record[k] for k in ('frame_count','fps','height','width','pair_id')):
                    raise ValueError('GT metadata differs')
                fidelity = manifest['pixel_supervision_fidelity']
                masks = {m['frame_index']:m for m in manifest['effect_masks']}
                if len(masks) != len(manifest['effect_masks']) or set(masks) != set(np.flatnonzero(targets)) or set(masks) != set(manifest['nonzero_mask_frames']):
                    raise ValueError('Sparse native GT/event coverage differs')
            previous = None
            local = []
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            feature_name, oracle_name = f'features_{video_index:03d}.jsonl', f'oracle_{video_index:03d}.jsonl'
            with torch.inference_mode(), (out/feature_name).open('x') as features_file, (out/oracle_name).open('x') as oracle_file:
                for t, rgb, timing in native_frames(source):
                    current = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0).cuda()
                    entry = {'sample_id':source['sample_id'], 'frame_index':t, **timing, 'no_predecessor':previous is None, 'features':None}
                    evidence_maps = None
                    if previous is not None:
                        x, y, padding = prepare_raft_pair(current, previous)
                        model.corr_block.corr_pyramid = []
                        backward = padding.unpad(model(x,y,num_flow_updates=12)[-1]).contiguous()
                        model.corr_block.corr_pyramid = []
                        forward = padding.unpad(model(y,x,num_flow_updates=12)[-1]).contiguous()
                        model.corr_block.corr_pyramid = []
                        evidence_maps = native_flow_evidence(previous,current,backward,forward)
                        entry['features'] = {'raw':residual_statistics(evidence_maps['raw_residual']),
                                             'aligned':residual_statistics(evidence_maps['aligned_residual'])}
                        for key in ('sampling_support','roundtrip_support','cycle_error_px','motion_magnitude_px'):
                            value = evidence_maps[key].float()
                            if not torch.isfinite(value).all():
                                raise ValueError('Non-finite flow quality evidence')
                            entry['features'][key] = {'mean':value.mean().item(),'max':value.max().item()}
                        if t == source['frame_count']//2:
                            np.savez_compressed(out/f'midpoint_{video_index:03d}.npz',previous_rgb=previous.cpu().numpy(),
                                current_rgb=current.cpu().numpy(),current_to_previous=backward.cpu().numpy(),
                                previous_to_current=forward.cpu().numpy(),**{k:v.cpu().numpy() for k,v in evidence_maps.items()})
                        del x,y,backward,forward
                    
                    mask = np.zeros((source['height'],source['width']),dtype=bool)
                    if t in masks:
                        item = masks[t]
                        bind(item['path'],item['sha256'])
                        with Image.open(item['path']) as image:
                            mask = np.asarray(image.convert('L')) > 0
                        if mask.shape != (source['height'],source['width']) or int(mask.sum()) != item['changed_pixel_count']:
                            raise ValueError('Native oracle pixels differ')
                    if bool(mask.any()) != bool(targets[t]):
                        raise ValueError('Oracle time/pixel labels disagree')
                    oracle = {'sample_id':source['sample_id'],'frame_index':t,'target':bool(targets[t]),
                              'phase':phase_at(targets,t),'fidelity':fidelity,'oracle_regions':None}
                    if evidence_maps is not None:
                        oracle['oracle_regions'] = region_sums(evidence_maps['raw_residual'], evidence_maps['aligned_residual'],
                                                             torch.from_numpy(mask[None]).cuda())
                    features_file.write(json.dumps(entry,allow_nan=False)+'\n')
                    oracle_file.write(json.dumps(oracle,allow_nan=False)+'\n')
                    features_file.flush(); oracle_file.flush()
                    merged = {**entry, **oracle}
                    local.append(merged)
                    category = record.get('hard_negative_category','not_hn')
                    group = '|'.join((record['role'],common.root_of(record),record['camera_condition'],category,oracle['phase'],fidelity))
                    groups[group].append(merged)
                    previous = current
                    del evidence_maps
                    update(status='native_train_signal_extraction',video_index=video_index+1,total_videos=300,
                           sample_id=source['sample_id'],source_frame_end=t+1,total_source_frames=source['frame_count'])
            torch.cuda.synchronize()
            bind(source['video_path'],source['video_sha256'])
            if len(local) != source['frame_count'] or [r['frame_index'] for r in local] != list(range(len(local))):
                raise ValueError('Incomplete feature/oracle coverage')
            item = {'sample_id':source['sample_id'],'scene_id':record['scene_id'],'role':record['role'],
                    'frame_count':len(local),'video_sha256':source['video_sha256'],
                    'features_path':str(out/feature_name),'features_sha256':sha(out/feature_name),
                    'oracle_path':str(out/oracle_name),'oracle_sha256':sha(out/oracle_name),
                    'midpoint_path':str(out/f'midpoint_{video_index:03d}.npz'),
                    'midpoint_sha256':sha(out/f'midpoint_{video_index:03d}.npz'),
                    'elapsed_seconds':time.monotonic()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()}
            inventory.append(item)
            cases.append({'sample_id':source['sample_id'],'role':record['role'],'root':common.root_of(record),
                          'camera_condition':record['camera_condition'],'fidelity':fidelity,**reduce_group(local)})
            save(out/'inventory.json',inventory)
            save(out/'input_hashes.json',hashes)
            del previous,current,local
        for path,expected in list(hashes.items()):
            bind(path,expected)
        save(out/'train_diagnostic.json',{'complete':True,'efficacy_evaluated':False,'threshold_fitted':False,
             'train_videos':300,'source_frames':sum(r['frame_count'] for r in inventory),
             'development_used':False,'test_used':False,'real_gt_used':False,
             'groups':{key:reduce_group(value) for key,value in sorted(groups.items())},'cases':cases})
        update(status='completed',completed_utc=common.utc_now())
    except Exception as exc:
        update(status='failed',error=repr(exc))
        raise


if __name__ == '__main__':
    main()
