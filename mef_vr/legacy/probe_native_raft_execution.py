

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import av
import numpy as np
import torch
import torchvision
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from native_dense_flow_evidence import prepare_raft_pair, native_flow_evidence


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def select_triple(rows, formal):
    pair = formal['pair_order'][0]
    targets = [('bug', lambda r: r['pair_id'] == pair),
               ('clean_or_fixed', lambda r: r['pair_id'] == pair),
               ('hard_negative', lambda r: r['sample_id'] == formal['hn_pairing'][pair])]
    selected = []
    for role, match in targets:
        found = [r for r in rows if r['role'] == role and match(r)]
        if len(found) != 1 or found[0]['split'] != 'train':
            raise ValueError('Require the unique pre-existing Train triple')
        selected.append(found[0])
    if selected[0]['scene_id'] != selected[1]['scene_id']:
        raise ValueError('Paired scene differs')
    return selected


def native_frames(record):
    
    with av.open(record['video_path']) as c:
        stream = c.streams.video[0]
        stream.thread_count = 2
        if abs(float(stream.average_rate) - record['fps']) > 1e-4:
            raise ValueError('Source FPS mismatch')
        previous_time = None
        count = 0
        for frame in c.decode(video=0):
            rgb = frame.to_ndarray(format='rgb24')
            if rgb.shape != (record['height'], record['width'], 3) or frame.pts is None:
                raise ValueError('Source native grid or PTS mismatch')
            timestamp = float(frame.pts * frame.time_base)
            if previous_time is not None and abs(timestamp - previous_time - 1/record['fps']) > 1e-4:
                raise ValueError('Nonuniform or missing source frame')
            yield count, rgb, {'source_time_s': timestamp, 'pts': frame.pts,
                               'time_base': str(frame.time_base)}
            previous_time = timestamp
            count += 1
        if count != record['frame_count']:
            raise ValueError('Incomplete source video')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--formal-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    out = a.output
    if not a.worker:
        out.mkdir(parents=True, exist_ok=False)
        (out/'code').mkdir()
        for name in [Path(__file__).name, 'native_dense_flow_evidence.py']:
            shutil.copy2(Path(__file__).with_name(name), out/'code'/name)
        os.execv(sys.executable, [sys.executable, str(out/'code'/Path(__file__).name), *sys.argv[1:], '--worker'])
    state = {'pid': os.getpid(), 'status': 'preparing', 'optimizer_steps': 0,
             'started_utc': datetime.now(timezone.utc).isoformat()}
    def update(**kw):
        state.update(kw)
        save(out/'status.json', state)
    hashes = {}
    def bind(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('Hash mismatch: ' + str(path))
        hashes[str(path)] = actual
        return actual
    update()
    try:
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12*1024**3:
            raise RuntimeError('Need >=12GiB free; do not change active training')
        torch.cuda.set_per_process_memory_fraction(.25)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        bind(a.formal_run/'protocol.json')
        formal = json.loads((a.formal_run/'protocol.json').read_text())
        bind(a.evidence/'protocol.json')
        evidence = json.loads((a.evidence/'protocol.json').read_text())
        rows = []
        for cohort in ['train_paired', 'train_hn']:
            mp = a.evidence/evidence['cohort_manifests'][cohort]
            bind(mp, evidence['output_sha256'][mp.name])
            rows += [json.loads(line) for line in mp.read_text().splitlines() if line.strip()]
        if any(r['split'] != 'train' for r in rows):
            raise ValueError('Non-Train input manifest')
        triple = select_triple(rows, formal)
        weights = Raft_Large_Weights.C_T_SKHT_V2
        protocol = {
            'purpose': 'full-native execution feasibility only; no classifier, optimization or efficacy estimate',
            'selection': 'first existing formal Train triple; no outcome-based selection',
            'weights': weights.name, 'weights_url': weights.url, 'iterations': 12,
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'dtype': 'float32', 'tf32': False, 'memory_fraction_cap': .25,
            'required_free_gib': 12, 'bidirectional': True,
            'all_source_frames': True, 'resize': False, 'fps_changed': False,
            'mask_or_paired_clean_in_features': False, 'gt_files_opened': False,
            'test_used': False, 'development_used': False, 'real_gt_used': False,
            'current_training_changed': False, 'optimizer_steps': 0,
            'comparison_warning': 'Residual summary is NOT a detection metric; concurrent GPU time is not isolated speed',
            'saved_outputs': 'per-frame scalar statistics and PTS; one fixed midpoint native flow evidence NPZ per video, not all dense flows',
            'sources': [{k: r[k] for k in ['sample_id','split','role','video_path','video_sha256',
                         'frame_count','fps','height','width','scene_id']} for r in triple],
            'code_sha256': {f.name: sha(f) for f in (out/'code').glob('*.py')},
            'dependency_sha256': {name: sha(Path(torchvision.__file__).parent/name)
                                 for name in ['models/optical_flow/raft.py','transforms/_presets.py']},
        }
        save(out/'protocol.json', protocol)
        update(status='downloading_official_weights')
        weight_path = out/'raft_large_C_T_SKHT_V2-ff5fadd5.pth'
        torch.hub.download_url_to_file(weights.url, str(weight_path), hash_prefix='ff5fadd5', progress=False)
        protocol['weights_sha256'] = bind(weight_path)
        save(out/'protocol.json', protocol)
        model = raft_large(weights=None, progress=False)
        model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True), strict=True)
        model = model.cuda().eval()
        model.requires_grad_(False)
        results = []
        for source_index, record in enumerate(triple):
            bind(record['video_path'], record['video_sha256'])
            previous = None
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            stats = []
            with torch.inference_mode(), (out/f'frames_{source_index}.jsonl').open('x') as log:
                for index, rgb, timing in native_frames(record):
                    current = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).cuda()
                    entry = {'sample_id': record['sample_id'], 'frame_index': index,
                             'no_predecessor': previous is None, **timing}
                    if previous is None:
                        entry['features'] = None  
                    else:
                        x, y, pad = prepare_raft_pair(current, previous)
                        
                        model.corr_block.corr_pyramid = []
                        back = pad.unpad(model(x, y, num_flow_updates=12)[-1]).contiguous()
                        model.corr_block.corr_pyramid = []
                        forward = pad.unpad(model(y, x, num_flow_updates=12)[-1]).contiguous()
                        model.corr_block.corr_pyramid = []
                        evidence_maps = native_flow_evidence(previous, current, back, forward)
                        if any(not torch.isfinite(v).all() for v in evidence_maps.values()):
                            raise ValueError('Non-finite native evidence')
                        entry['features'] = {key: {'mean': v.float().mean().item(), 'max': v.float().max().item()}
                                             for key, v in evidence_maps.items()}
                        if index == record['frame_count']//2:
                            np.savez_compressed(out/f'midpoint_{source_index}.npz',
                                previous_rgb=previous.cpu().numpy(), current_rgb=current.cpu().numpy(),
                                current_to_previous=back.cpu().numpy(), previous_to_current=forward.cpu().numpy(),
                                **{key: v.cpu().numpy() for key, v in evidence_maps.items()})
                        del x, y, back, forward, evidence_maps
                    log.write(json.dumps(entry, allow_nan=False) + '\n')
                    log.flush()
                    stats.append(entry)
                    previous = current
                    update(status='native_bidirectional_inference', sample_id=record['sample_id'],
                           source_index=source_index, source_frame_end=index+1, total_source_frames=record['frame_count'])
            torch.cuda.synchronize()
            bind(record['video_path'], record['video_sha256'])
            if len(stats) != record['frame_count'] or [s['frame_index'] for s in stats] != list(range(record['frame_count'])):
                raise ValueError('Output frame coverage differs')
            results.append({'sample_id': record['sample_id'], 'source_frames': len(stats),
                'directed_flow_passes': 2*(len(stats)-1), 'elapsed_seconds': time.monotonic()-started,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                'original_first_pts': stats[0]['pts'], 'original_last_pts': stats[-1]['pts'],
                'output_sha256': {n: sha(out/n) for n in [f'frames_{source_index}.jsonl', f'midpoint_{source_index}.npz']}})
            save(out/'completed_sources.json', results)
            del previous, current
        for path, expected in list(hashes.items()):
            bind(path, expected)
        save(out/'input_hashes.json', hashes)
        save(out/'report.json', {'execution_complete': True, 'efficacy_evaluated': False,
             'optimizer_steps': 0, 'source_frames': sum(r['source_frames'] for r in results),
             'directed_flow_passes': sum(r['directed_flow_passes'] for r in results), 'results': results})
        update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        update(status='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
