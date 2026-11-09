
from __future__ import annotations
from fractions import Fraction
import numpy as np

from report_native_tolerant_events import intervals
from r3d_native_transfer_scoring import require_packed


def native_box(mask):
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError('Native boolean H,W prediction required')
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)] if len(xs) else None


def output_prediction(observation, frame_scores, native_predictions):
    
    prohibited = {'gt', 'target', 'mask_gt', 'roi', 'clean_path', 'gt_label', 'root_cause'}
    if set(observation) & prohibited: raise ValueError('Annotation-free observation metadata required')
    n, h, w = (observation[k] for k in ('frame_count', 'height', 'width'))
    if any(type(v) is not int or v <= 0 for v in (n, h, w)):
        raise ValueError('Positive original geometry required')
    require_packed(native_predictions, (n, h, w))
    score = np.asarray(frame_scores, np.float64)
    pts = observation['source_pts']; times = np.asarray(observation['source_time_s'], np.float64)
    if (score.shape != (n,) or times.shape != (n,) or len(pts) != n or not np.isfinite(score).all()
            or ((score < 0) | (score > 1)).any() or not np.isfinite(times).all()
            or any(type(p) is not int for p in pts) or any(b <= a for a, b in zip(pts, pts[1:]))):
        raise ValueError('Complete finite source scores and strictly increasing original clock required')
    base = Fraction(observation['time_base_fraction'])
    if base <= 0 or not np.allclose(times, [float(p*base) for p in pts], rtol=0, atol=1e-9):
        raise ValueError('Source timestamps and original PTS disagree')
    start = observation['source_start_frame']
    if type(start) is not int or start < 0 or observation['source_end_frame_exclusive'] != start+n:
        raise ValueError('Original source frame interval differs')
    keep = score >= .5
    masks = np.array(native_predictions, copy=True); masks[~keep] = 0
    frames = []
    for t in range(n):
        mask = np.unpackbits(masks[t], count=h*w, bitorder='little').reshape(h, w).astype(bool)
        area = int(mask.sum())
        frames.append({'source_frame_index': start+t, 'source_pts': pts[t], 'source_time_s': float(times[t]),
            'flicker_score': float(score[t]), 'flicker_predicted': bool(keep[t]),
            'predicted_pixels': area, 'bbox_xyxy': native_box(mask),
            'spatial_prediction_missing': bool(keep[t] and not area)})
    events = []
    for a, b in intervals(keep.tolist()):
        union = np.bitwise_or.reduce(masks[a:b+1], axis=0)
        union = np.unpackbits(union, count=h*w, bitorder='little').reshape(h, w).astype(bool)
        events.append({'start_frame_inclusive': start+a, 'end_frame_inclusive': start+b,
            'start_frame_time_s': float(times[a]), 'end_frame_time_s': float(times[b]),
            'positive_frame_count': b-a+1, 'max_frame_score': float(score[a:b+1].max()),
            'bbox_xyxy_union': native_box(union), 'spatial_prediction_missing': not bool(union.any())})
    return {'schema_version': '1.0',
        'input': {k: observation[k] for k in ('video_path', 'video_sha256', 'frame_count', 'height', 'width',
            'fps_fraction', 'time_base_fraction', 'source_start_frame', 'source_end_frame_exclusive', 'approved_seconds')},
        'flicker_predicted': bool(keep.any()), 'video_max_frame_score': float(score.max()),
        'threshold': .5, 'score_is_calibrated_probability': False,
        'frame_indexing': 'zero_based_original_source; inclusive event endpoints',
        'event_time_semantics': 'Timestamp of first/last positive original frame, NOT an inferred end-of-frame time',
        'bbox_convention': 'Original native pixels: x0,y0 inclusive; x1,y1 exclusive',
        'events': events, 'frames': frames,
        'spatial_method': 'Original frozen C predictions gated by frozen parent_DINO time; no new pixel decoder',
        'root_cause_prediction': None, 'root_cause_note': 'Not determined by this detection entry point',
        'GT_used': False, 'paired_Clean_used': False, 'accuracy_computed': False,
        'source_resized': False, 'source_fps_changed': False, 'source_frames_dropped': False,
        'backend': 'experimental_CPU_not_claimed_numerically_equivalent_to_GPU',
        'real_world_performance_validated': False}, masks
