
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from run_flicker_multimodel_pilot import save, sha

STATE_HEAD_HALO = 6  


def scan_observed_video(path):
    import av
    pts, dimensions, base, rate = [], None, None, None
    with av.open(str(path)) as c:
        if len(c.streams.video) != 1: raise ValueError('One unambiguous video stream required')
        stream = c.streams.video[0]; stream.thread_count = 2
        if not stream.average_rate or not stream.time_base: raise ValueError('Missing original clock')
        rate, base = Fraction(stream.average_rate), Fraction(stream.time_base)
        for frame in c.decode(video=0):
            shape = (frame.height, frame.width)
            if dimensions is None: dimensions = shape
            if (shape != dimensions or frame.pts is None or Fraction(frame.time_base) != base
                    or (pts and frame.pts <= pts[-1])):
                raise ValueError('Changing geometry, missing or nonmonotonic original PTS')
            pts.append(int(frame.pts))
    if not pts or rate <= 0: raise ValueError('Empty or invalid source')
    times = [float(t * base) for t in pts]
    try:
        from r3d_native_video_store import validate_source_clock
        validate_source_clock(pts, base, rate); cfr = True
    except ValueError: cfr = False
    return {'frame_count': len(pts), 'height': dimensions[0], 'width': dimensions[1],
            'fps': float(rate), 'fps_fraction': str(rate), 'time_base_fraction': str(base),
            'source_pts': pts, 'source_time_s': times, 'cfr_compatible': cfr,
            'clock_policy': 'Every decoded original frame/PTS; never resampled or reordered'}


def approved_frame_interval(clock, approved_seconds=None):
    
    n = clock['frame_count']
    if approved_seconds is None: return 0, n
    if (len(approved_seconds) != 2 or not all(math.isfinite(v) for v in approved_seconds)
            or not 0 <= approved_seconds[0] < approved_seconds[1]):
        raise ValueError('Explicit finite increasing approved interval required')
    start = max(0, math.floor(approved_seconds[0] * clock['fps'] + 1e-9))
    end = min(n, math.ceil(approved_seconds[1] * clock['fps'] - 1e-9))
    if not start < end: raise ValueError('Approved range contains no original frames')
    return start, end


def create_observed_store(path, out, expected_sha256, approved_seconds=None, reserve_bytes=40 * 1024**3):
    
    import av
    path, out = Path(path).resolve(strict=True), Path(out).resolve()
    if out.exists(): raise FileExistsError('Keep earlier source stores')
    digest = sha(path)
    if digest != expected_sha256: raise ValueError('Unexpected input video')
    clock = scan_observed_video(path)
    start, end = approved_frame_interval(clock, approved_seconds)
    shape = (end - start, clock['height'], clock['width'], 3)
    if shutil.disk_usage(out.parent).free < int(np.prod(shape)) + reserve_bytes + 1024**2:
        raise OSError('Insufficient space above reserve; do not delete old data')
    out.mkdir()
    save(out / 'status.json', {'status': 'decoding', 'source': str(path)})
    rgb = np.lib.format.open_memmap(out / 'rgb.npy', mode='w+', dtype=np.uint8, shape=shape)
    decoded = hashlib.sha256(); seen = 0; retained = 0
    try:
        with av.open(str(path)) as c:
            stream = c.streams.video[0]; stream.thread_count = 2
            for t, frame in enumerate(c.decode(video=0)):
                if (t >= clock['frame_count'] or frame.pts != clock['source_pts'][t]
                        or (frame.height, frame.width) != shape[1:3]):
                    raise ValueError('Second decode differs from original source scan')
                seen += 1
                if start <= t < end:
                    pixels = frame.to_ndarray(format='rgb24')
                    rgb[t - start] = pixels; decoded.update(memoryview(np.ascontiguousarray(pixels))); retained += 1
                    if retained % 32 == 0: rgb.flush()
        rgb.flush()
        if seen != clock['frame_count'] or retained != end - start or sha(path) != digest:
            raise ValueError('Incomplete or changed original video')
        observation = {'video_path': str(path), 'video_sha256': digest,
            'source_frame_count': clock['frame_count'], 'source_start_frame': start, 'source_end_frame_exclusive': end,
            'frame_count': end - start, 'height': clock['height'], 'width': clock['width'], 'fps': clock['fps'],
            'fps_fraction': clock['fps_fraction'], 'time_base_fraction': clock['time_base_fraction'],
            'source_pts': clock['source_pts'][start:end], 'source_time_s': clock['source_time_s'][start:end],
            'cfr_compatible': clock['cfr_compatible'], 'approved_seconds': approved_seconds,
            'rgb_sha256': decoded.hexdigest(), 'rgb_npy_sha256': sha(out / 'rgb.npy'),
            'spatial_resize': False, 'temporal_sampling': False, 'fps_resampled': False, 'video_reencoded': False,
            'labels_loaded': False, 'paired_clean_used': False,
            'vfr_caveat': 'Model frame lags stay in native frame units; accepting VFR does not prove rate invariance'}
        save(out / 'observation.json', observation)
        save(out / 'status.json', {'status': 'completed', 'frames': retained})
        return observation
    except BaseException as exc:
        save(out / 'status.json', {'status': 'failed', 'error': str(exc)}); raise
    finally:
        del rgb


def dense_prediction_chunks(model, features, mean, std, *, core_frames=32, device='cuda'):
    
    import torch
    from flicker_multimodel_core import DenseStateHead
    from flicker_multimodel_fast_state import FastResidualStateHead
    if type(model) not in (DenseStateHead, FastResidualStateHead) or any(m.training for m in model.modules()):
        raise ValueError('Known eval-only dense architecture required')
    if type(core_frames) is not int or core_frames < 1 or features.ndim != 4 or min(features.shape) < 1:
        raise ValueError('Complete T,C,h,w features and positive core length required')
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if (mean.shape != (features.shape[1],) or std.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()): raise ValueError('Frozen finite normalization required')
    mu = torch.from_numpy(mean.copy()).to(device)[None, :, None, None]
    sd = torch.from_numpy(std.copy()).to(device)[None, :, None, None]
    n = len(features)
    for start in range(0, n, core_frames):
        stop = min(n, start + core_frames)
        left, right = max(0, start - STATE_HEAD_HALO), min(n, stop + STATE_HEAD_HALO)
        data = np.array(features[left:right], copy=True)
        if not np.isfinite(data).all(): raise ValueError('Nonfinite observation features')
        with torch.inference_mode():
            x = (torch.from_numpy(data).to(device).float() - mu) / sd
            result = model(x)
            a, b = start - left, stop - left
            scores = result['frame_logits'][a:b].sigmoid().cpu().numpy()
            boundary = result['boundary_logits'][a:b].sigmoid().cpu().numpy()
        yield {'start_frame': start, 'end_frame_exclusive': stop,
               'frame_probabilities': scores, 'boundary_probabilities': boundary}


def complete_dense_prediction(model, features, mean, std, *, core_frames=32, device='cuda'):
    scores = np.empty(len(features), np.float32); boundaries = np.empty((len(features), 2), np.float32)
    seen = 0
    for row in dense_prediction_chunks(model, features, mean, std, core_frames=core_frames, device=device):
        first, last = row['start_frame'], row['end_frame_exclusive']
        if first != seen: raise ValueError('Missing or duplicated original output frame')
        scores[first:last] = row['frame_probabilities']; boundaries[first:last] = row['boundary_probabilities']; seen = last
    if seen != len(features): raise ValueError('Incomplete model prediction')
    return {'frame_probabilities': scores, 'boundary_probabilities': boundaries}
