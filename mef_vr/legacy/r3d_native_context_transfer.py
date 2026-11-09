
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np

from prepare_r3d_audited_native_context import require_decoded_observation
from r3d_native_video_store import validate_source_clock
from r3d_upstream_scope_audit_contract import coverage, require
from run_cache_encoding_consistent_control import require_baseline_scores, sha

ARMS = ('A_head_only', 'B_early_r3d')
SEEDS = ('20261005', '20261105')
PHASE_COUNTS = {'train': 300, 'development': 200}
OBSERVATION_KEYS = {
    'sample_id', 'video_path', 'video_sha256', 'frame_count', 'height',
    'width', 'fps', 'rgb_sha256', 'input_mode', 'clock',
}


def validate_prepared(item):
    
    require(set(item) == {'observation', 'context'}, 'Observation/context only')
    obs, context = item['observation'], item['context']
    require(set(obs) == OBSERVATION_KEYS, 'Explicit observation whitelist required')
    require(set(context) == {'sample_id', 'frame_probabilities', 'coverage'}, 'Explicit context whitelist required')
    require(obs['input_mode'] == 'original_mp4', 'Original encoded source required')
    require(isinstance(obs['sample_id'], str) and bool(obs['sample_id']), 'Opaque source ID required')
    require(context['sample_id'] == obs['sample_id'], 'Context ID differs')
    n, h, w = (obs[k] for k in ('frame_count', 'height', 'width'))
    require(all(type(v) is int and v > 0 for v in (n, h, w)), 'Native dimensions required')
    for key in ('video_sha256', 'rgb_sha256'):
        value = obs[key]
        require(isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value), 'Invalid source hash')
    require(isinstance(obs['video_path'], str) and Path(obs['video_path']).is_absolute(), 'Absolute source path required')
    clock = obs['clock']
    require(set(clock) == {'fps_fraction', 'time_base_fraction', 'source_pts', 'source_time_s'}, 'Original clock whitelist required')
    validate_source_clock(clock['source_pts'], clock['time_base_fraction'], clock['fps_fraction'])
    times = [float(Fraction(t) * Fraction(clock['time_base_fraction'])) for t in clock['source_pts']]
    require(len(times) == n and times == clock['source_time_s'], 'Absolute source clock differs')
    require(np.isfinite(obs['fps']) and abs(float(Fraction(clock['fps_fraction'])) - obs['fps']) < 1e-4, 'FPS differs')
    scores = np.asarray(context['frame_probabilities'], np.float64)
    require(scores.shape == (n,) and np.isfinite(scores).all()
            and ((scores >= 0) & (scores <= 1)).all(), 'Complete finite raw probabilities required')
    require(context['coverage'] == coverage(n)[1], 'Original32/8 coverage required')
    return obs['sample_id']


def join_contexts(by_seed, *, phase_counts=None):
    
    counts = PHASE_COUNTS if phase_counts is None else phase_counts
    require(set(counts) == set(PHASE_COUNTS)
            and all(type(v) is int and v > 0 for v in counts.values()), 'Train/development required')
    require(set(by_seed) == set(SEEDS), 'Exactly both preselected seeds required')
    joined = {phase: {} for phase in counts}
    for seed in SEEDS:
        require(set(by_seed[seed]) == set(counts), 'Complete paired phases required')
        for phase, count in counts.items():
            source = by_seed[seed][phase]
            require(set(source) == set(ARMS), 'Both arms required; no best-arm selection')
            for arm in ARMS:
                rows = source[arm]
                ids = [validate_prepared(row) for row in rows]
                require(len(ids) == len(set(ids)) == count, 'Complete unique source cohort required')
                if joined[phase]:
                    require(set(ids) == set(joined[phase]), 'Changed source population')
                for row in rows:
                    sid = row['observation']['sample_id']
                    if sid not in joined[phase]:
                        joined[phase][sid] = {'observation': copy.deepcopy(row['observation']), 'contexts': {}}
                    result = joined[phase][sid]
                    require(result['observation'] == row['observation'], 'RGB/clock identity differs between arms or seeds')
                    result['contexts'][seed + '__' + arm] = copy.deepcopy(row['context'])
    train, dev = joined['train'], joined['development']
    require(not set(train) & set(dev), 'Source ID crosses current-stage split')
    for key in ('video_path', 'video_sha256'):
        require(not {r['observation'][key] for r in train.values()}
                & {r['observation'][key] for r in dev.values()}, 'Original source crosses current-stage split')
    return {phase: [rows[sid] for sid in sorted(rows)] for phase, rows in joined.items()}


def observation_decode(obs):
    
    import av
    from r3d_native_video_store import scan_source
    require(sha(obs['video_path']) == obs['video_sha256'], 'Original video hash differs')
    clock = scan_source(obs['video_path'])
    n, h, w = (obs[k] for k in ('frame_count', 'height', 'width'))
    require((clock['frame_count'], clock['height'], clock['width'], clock['fps']) == (n, h, w, obs['fps']), 'Original shape/FPS differs')
    rgb = np.empty((n, h, w, 3), np.uint8)
    seen = 0
    with av.open(obs['video_path']) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        require(str(Fraction(stream.average_rate)) == clock['fps_fraction']
                and str(Fraction(stream.time_base)) == clock['time_base_fraction'], 'Decode clock changed')
        for frame in container.decode(video=0):
            require(seen < n and frame.pts == clock['source_pts'][seen]
                    and frame.time is not None and float(frame.time) == clock['source_time_s'][seen], 'Missing/repeated/changed original PTS')
            pixels = frame.to_ndarray(format='rgb24')
            require(pixels.shape == (h, w, 3) and pixels.dtype == np.uint8, 'Native RGB required')
            rgb[seen] = pixels
            seen += 1
    require(seen == n and sha(obs['video_path']) == obs['video_sha256'], 'Incomplete/changed original source')
    require(hashlib.sha256(memoryview(rgb)).hexdigest() == obs['rgb_sha256'], 'Upstream/native decoded RGB differs')
    require({key: clock[key] for key in obs['clock']} == obs['clock'], 'Full source clock differs')
    return rgb, clock


def predict_observation(predictor, rgb, clock, prepared):
    
    from r3d_native_zero_motion_inference import collect_native_chunks
    validate_prepared(prepared)
    scores, times = require_decoded_observation(rgb, clock, prepared)
    chunks = predictor(rgb, scores, prepared['observation']['fps'], times)
    masks, maxima = collect_native_chunks(chunks, tuple(rgb.shape[:3]), times)
    
    require(np.array_equal(masks.any(axis=1), maxima >= .5), 'Native mask/score decision differs')
    return dict(packed_masks=masks, frame_probabilities=maxima, source_time_s=times.copy(),
                native_shape=tuple(rgb.shape[:3]), pixel_threshold=.5)


def require_original_replay(actual, reference):
    
    for result in (actual, reference):
        shape = result['native_shape']
        require(len(shape) == 3 and all(isinstance(v, (int, np.integer))
                and not isinstance(v, (bool, np.bool_)) and v > 0 for v in shape), 'Native replay dimensions required')
        n, h, w = (int(v) for v in shape)
        masks = result['packed_masks']
        require(isinstance(masks, np.ndarray) and masks.dtype == np.uint8
                and masks.shape == (n, (h * w + 7) // 8), 'Complete native packed masks required')
        require(result['pixel_threshold'] == .5, 'Original output threshold required')
        if h * w % 8:
            require(not (masks[:, -1] & (255 << (h * w % 8) & 255)).any(), 'Nonzero packed-mask padding')
        scores = np.asarray(result['frame_probabilities'])
        require(scores.shape == (n,) and np.isfinite(scores).all()
                and ((scores >= 0) & (scores <= 1)).all(), 'Complete original frame scores required')
        require(np.array_equal(masks.any(axis=1), scores >= .5), 'Replay mask/score decision differs')
        times = np.asarray(result['source_time_s'])
        require(times.shape == (n,) and np.issubdtype(times.dtype, np.floating) and np.isfinite(times).all()
                and not (np.diff(times) <= 0).any(), 'Complete original replay PTS required')
    require(tuple(actual['native_shape']) == tuple(reference['native_shape']), 'Original native shape differs')
    require(np.array_equal(actual['source_time_s'], reference['source_time_s']), 'Original absolute PTS differs')
    require(np.array_equal(actual['packed_masks'], reference['packed_masks']), 'Original C binary replay differs')
    difference = require_baseline_scores(actual['frame_probabilities'], reference['frame_probabilities'])
    return dict(passed=True, binary_masks_exact=True, absolute_pts_exact=True,
                frames=int(actual['native_shape'][0]), score_max_abs_error=difference,
                score_atol=1e-6, score_rtol=1e-6)


def save_observation_prediction(path, result):
    
    path = Path(path)
    require_original_replay(result, result)
    
    with path.open('xb') as stream:
        np.savez_compressed(stream, **result)
    return dict(path=str(path), sha256=sha(path), frames=int(result['native_shape'][0]))
