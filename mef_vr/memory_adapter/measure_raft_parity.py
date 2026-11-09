
import argparse
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import av
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow.raft import CorrBlock

from raft_memory_adapter import DEFAULT_CHUNK_BYTES, InplaceCorrBlock, RecomputedCorrBlock, replace_corr_block, sha

HERE = Path(__file__).resolve().parent
FROZEN = Path('/mnt/e/VRFlickerLab/model_evaluation/real_gt_three_models_20261014_v1/ours/frozen_code')
PROTOCOL = FROZEN.parent/'protocol.json'
LOCK = Path('/tmp/vrflicker_real_gt_three_models_20261014_gpu.lock')
INPUT = Path('/mnt/e/VRFlickerLab/real_video_collection/clean_gt_freeze_20261014_v1/inference_manifest.json')
ATOL = RTOL = 1e-5


def compare(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('Parity geometry/precision differs')
    diff = (a-b).abs()
    return dict(bit_exact=bool(torch.equal(a, b)), max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                within_tolerance=bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)))


def load_pair():
    doc = json.loads(INPUT.read_text())
    row = next(r for r in doc['records'] if r['sample_id'] == 'CLEAN_G01')
    assert sha(row['video_path']) == row['sha256']
    frames = []
    with av.open(row['video_path']) as container:
        stream = container.streams.video[0]; stream.thread_count = 1
        for frame in container.decode(video=0):
            if frame.pts in row['source_pts'][:2]:
                frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames) == 2: break
    assert len(frames) == 2 and frames[0].shape == (row['height'], row['width'], 3)
    return frames, dict(sample_id=row['sample_id'], video_path=row['video_path'], source_sha256=row['sha256'],
                       source_pts=row['source_pts'][:2], dimensions=[row['height'], row['width']],
                       input_manifest_sha256=sha(INPUT), resampled=False, cropped=False, labels_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'gpu_parity_receipt_v1.json')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve previous measurement')
    sys.path.insert(0, str(FROZEN))
    from native_dense_flow_evidence import prepare_raft_pair, native_flow_evidence
    from native_flow_consistency import bidirectional_consistency
    p = json.loads(PROTOCOL.read_text())
    weights = p['models']['raft']
    assert sha(weights) == p['model_bindings'][weights]
    frames, source = load_pair()
    started = time.monotonic()
    records = {}; corr_checks = {}
    print('Waiting for original shared GPU lock; no GPU work has started', flush=True)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print('Acquired original shared GPU lock for parity measurement', flush=True)
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.65)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(20261014)
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            
            a, b = [torch.randn(1, 256, 17, 19, device='cuda') for _ in range(2)]
            y, x = torch.meshgrid(torch.arange(17, device='cuda'), torch.arange(19, device='cuda'), indexing='ij')
            centers = torch.stack((x, y))[None].float() + .3125
            old = CorrBlock(); old.build_pyramid(a, b); reference = old.index_pyramid(centers)
            for mode, block in [('inplace', InplaceCorrBlock()), ('recompute', RecomputedCorrBlock(max_chunk_bytes=4*17*19*7))]:
                block.build_pyramid(a, b)
                corr_checks[mode] = compare(reference, block.index_pyramid(centers))
            del a, b, centers, old, block, reference
            model = raft_large(weights=None, progress=False)
            model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
            model = model.cuda().eval().requires_grad_(False)
            prev, curr = [torch.from_numpy(np.ascontiguousarray(f.transpose(2, 0, 1)))[None].cuda() for f in frames]
            x, y, pad = prepare_raft_pair(curr, prev)
            outputs = {}; evidence = {}
            for mode in ('original', 'inplace', 'recompute'):
                model.corr_block = CorrBlock()
                if mode != 'original': replace_corr_block(model, mode)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                elapsed = time.monotonic(); directions = []
                for first, second in ((x, y), (y, x)):
                    model.corr_block.corr_pyramid = []
                    predicted = model(first, second, num_flow_updates=12)
                    directions.append([pad.unpad(f).contiguous().cpu() for f in predicted])
                    del predicted
                    model.corr_block.corr_pyramid = []
                back, forward = directions[0][-1].cuda(), directions[1][-1].cuda()
                maps = native_flow_evidence(prev, curr, back, forward)
                flags = bidirectional_consistency(back, forward)
                evidence[mode] = {**{k: v.cpu() for k, v in maps.items()},
                                  **{'consistency_'+k: v.cpu() for k, v in flags.items()}}
                del back, forward, maps, flags
                torch.cuda.synchronize()
                records[mode] = dict(seconds=time.monotonic()-elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                     updates_per_direction=12, directions=2)
                outputs[mode] = directions
                if mode == 'recompute': records[mode]['chunk_stats'] = model.corr_block.stats
                print(json.dumps(dict(mode=mode, **records[mode])), flush=True)
            for mode in ('inplace', 'recompute'):
                records[mode]['flow_comparisons'] = [compare(a, b) for base, candidate in zip(outputs['original'], outputs[mode]) for a, b in zip(base, candidate)]
                stats = {}
                for key, value in evidence['original'].items():
                    candidate = evidence[mode][key]
                    stats[key] = dict(bit_exact=bool(torch.equal(value, candidate)), within_tolerance=bool(torch.equal(value, candidate))) if value.dtype == torch.bool else compare(value, candidate)
                records[mode]['evidence_comparisons'] = stats
                records[mode]['passed'] = corr_checks[mode]['within_tolerance'] and all(v['within_tolerance'] for v in records[mode]['flow_comparisons']) and all(v['within_tolerance'] for v in stats.values())
            del model, x, y, prev, curr
            torch.cuda.empty_cache()
    raft_module = importlib.import_module('torchvision.models.optical_flow.raft')
    passed_modes = [m for m in ('inplace', 'recompute') if records[m]['passed']]
    report = dict(passed=bool(passed_modes), passed_modes=passed_modes, created_utc=datetime.now(timezone.utc).isoformat(),
                  adapter_sha256=sha(HERE/'raft_memory_adapter.py'), measurement_script_sha256=sha(__file__),
                  torchvision_raft_sha256=sha(raft_module.__file__), frozen_producer_sha256=sha(FROZEN/'build_native_motion_train_cache.py'),
                  weights_path=weights, weights_sha256=sha(weights), max_chunk_bytes=DEFAULT_CHUNK_BYTES,
                  torch_version=torch.__version__, source=source, shared_gpu_lock=str(LOCK),
                  precision='FP32', tf32=False, memory_fraction=.65, atol=ATOL, rtol=RTOL,
                  random_correlation_checks=corr_checks, measurements=records, seconds=time.monotonic()-started,
                  limits='One complete native 720p adjacent-frame pair plus a chunk-tail correlation fixture; not proof on every future input; no model or threshold tuning.')
    with args.output.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(receipt=str(args.output), passed_modes=passed_modes, passed=report['passed'])), flush=True)


if __name__ == '__main__': main()
