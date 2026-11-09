
from __future__ import annotations
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import time
import numpy as np

from flicker_multimodel_core import pool_native
from run_flicker_multimodel_pilot import save, sha

REVIEW_RENDER_VERSION = 'native_predicted_pixels_no_occluding_banner_v2'


def overlay_predicted_pixels(rgb_frame, packed_mask, score):
    
    from r3d_native_transfer_scoring import require_packed
    if (rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3 or rgb_frame.dtype != np.uint8
            or min(rgb_frame.shape[:2]) < 1 or not np.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('Native uint8 RGB and finite unit-interval score required')
    h, w = rgb_frame.shape[:2]
    require_packed(np.asarray(packed_mask)[None], (1, h, w))
    image = np.array(rgb_frame, copy=True)
    if score >= .5:
        mask = np.unpackbits(packed_mask, count=h*w, bitorder='little').reshape(h, w).astype(bool)
        image[mask] = (image[mask].astype(np.float32)*.55 + np.array([255, 0, 0])*.45).astype(np.uint8)
    return image


def validate_sources(sources):
    keys = {'sample_id', 'video_path', 'video_sha256', 'approved_seconds'}
    if not sources or len({s['sample_id'] for s in sources}) != len(sources):
        raise ValueError('Distinct complete original observations required')
    for s in sources:
        if set(s) != keys: raise ValueError('Source descriptor must not contain label/GT/ROI information')
        if s['approved_seconds'] is not None:
            t = s['approved_seconds']
            if len(t) != 2 or not np.isfinite(t).all() or not 0 <= t[0] < t[1]:
                raise ValueError('Only original explicitly approved interval allowed')


def replace_upstream_features(features, names, scores, packed, shape):
    
    import torch
    n, h, w = shape; gh, gw = (h+13)//14, (w+13)//14
    if (features.shape != (n, len(names), gh, gw) or features.dtype != np.float16
            or packed.shape != (n, (h*w+7)//8) or packed.dtype != np.uint8
            or scores.shape != (n,) or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any() or len(set(names)) != len(names)):
        raise ValueError('Complete native same-coordinate observations required')
    result = np.array(features, copy=True)
    ci = names.index('frozen_C_mask_fraction'); hi = names.index('frozen_r3d_context')
    result[:, hi] = scores.astype(np.float32)[:, None, None]
    for t in range(n):
        mask = np.unpackbits(packed[t], count=h*w, bitorder='little').reshape(h, w)
        result[t, ci] = pool_native(torch.from_numpy(mask.copy())[None, None].float())[0, 0].numpy()
    if not np.isfinite(result).all(): raise ValueError('Nonfinite full CPU features')
    return result


def binary_diagnostic(prediction_rows, label_rows, expected_ids):
    
    expected = set(expected_ids)
    if len(expected) != len(expected_ids) or set(label_rows) != expected:
        raise ValueError('Frozen full historical cohort labels required')
    seen = set(); counters = {}; details = []
    for sid, predictions in prediction_rows:
        if sid not in expected or sid in seen: raise ValueError('Changed/repeated historical prediction source')
        seen.add(sid)
        for arm, row in predictions.items():
            p = np.asarray(row['frame_probabilities'], np.float64)
            if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError('Complete finite original-frame scores required')
            if type(label_rows[sid]) is not bool: raise ValueError('Explicit historical boolean labels required')
            truth = label_rows[sid]; score = float(p.max()); decision = score >= .5
            counts = counters.setdefault(arm, Counter())
            counts['tp' if truth and decision else 'fp' if decision else 'fn' if truth else 'tn'] += 1
            details.append({'sample_id': sid, 'arm': arm, 'actual': truth, 'prediction': decision, 'score': score})
    metrics = {}
    for arm, c in counters.items():
        tp, fp, fn, tn = (c[k] for k in ('tp', 'fp', 'fn', 'tn')); n = tp+fp+fn+tn
        if n != len(seen): raise ValueError('All frozen candidate arms required on all completed sources')
        metrics[arm] = dict(tp=tp, fp=fp, fn=fn, tn=tn, scored_videos=n, accuracy=(tp+tn)/n,
            recall=tp/(tp+fn) if tp+fn else None, specificity=tn/(tn+fp) if tn+fp else None,
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return {'metrics': metrics, 'details': details, 'expected_videos': len(expected), 'scored_videos': len(seen),
            'missing_ids': sorted(expected-seen), 'completed': seen == expected, 'new_holdout': False,
            'historical_real_reused': True, 'threshold_fitted_to_real': False, 'real_frame_pixel_localization_scored': False}


def review_cpu(folder, observation, rgb, packed, scores, binding):
    
    import av
    folder = Path(folder); dest = folder/'parent_DINO_CPU_review.mp4'; meta = dest.with_suffix('.json')
    if meta.exists():
        m = json.loads(meta.read_text())
        if m['binding'] != binding or sha(dest) != m['sha256']: raise ValueError('Committed review changed')
        if m.get('render_version') != REVIEW_RENDER_VERSION:
            raise ValueError('Preserve earlier renderer output; use a fresh review folder')
        return str(dest)
    if dest.exists(): dest.rename(dest.with_name(dest.name+f'.uncommitted_{time.time_ns()}'))
    obs = observation; tmp = dest.with_name(dest.stem+'.tmp.mp4')
    if tmp.exists(): tmp.rename(tmp.with_name(tmp.name+f'.uncommitted_{time.time_ns()}'))
    if len(rgb) != len(scores) or len(rgb) != obs['frame_count']: raise ValueError('Complete original review frames required')
    with av.open(str(tmp), 'w') as container:
        stream = container.add_stream('libx264', rate=Fraction(obs['fps_fraction']))
        stream.width, stream.height = obs['width'], obs['height']
        
        
        stream.pix_fmt = 'yuv444p' if obs['width'] % 2 or obs['height'] % 2 else 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast', 'threads': '2'}
        stream.time_base = Fraction(obs['time_base_fraction']); stream.codec_context.time_base = stream.time_base
        for t in range(len(rgb)):
            image = overlay_predicted_pixels(rgb[t], packed[t], scores[t])
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            frame.pts = obs['source_pts'][t]-obs['source_pts'][0]; frame.time_base = Fraction(obs['time_base_fraction'])
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    with av.open(str(tmp)) as container:
        times = []
        for frame in container.decode(video=0):
            if (frame.height, frame.width) != (obs['height'], obs['width']):
                raise ValueError('Review must preserve original pixel geometry')
            times.append(float(frame.pts*frame.time_base))
    expected = np.asarray(obs['source_time_s'])-obs['source_time_s'][0]
    if len(times) != len(expected) or not np.allclose(times, expected, rtol=0, atol=2e-5):
        raise ValueError('Review must preserve every original PTS interval')
    tmp.replace(dest)
    save(meta, {'binding': binding, 'sha256': sha(dest), 'original_frames': len(times),
                'all_original_pts_intervals_preserved': True, 'GT_overlay': False,
                'render_version': REVIEW_RENDER_VERSION, 'text_or_banner_occlusion': False,
                'overlay_semantics': 'Frozen C predicted pixels gated by parent time, not GT',
                'source_start_frame': obs['source_start_frame'], 'source_pts': obs['source_pts'],
                'source_time_s': obs['source_time_s'], 'frame_scores': [float(p) for p in scores],
                'codec_caveat': 'Review is lossy encoded visualization; packed native prediction mask is authoritative'})
    return str(dest)
