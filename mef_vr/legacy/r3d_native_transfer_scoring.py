
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from r3d_native_context_transfer import require_original_replay
from r3d_upstream_scope_audit_contract import frame_labels, require
from r3d_native_pixel_metrics import (
    boundary_counts, boundary_metrics, count_metrics, summarize_frames, unpack_mask,
)
from report_native_detail_subgroups import spatial_quality, temporal_strata, validate_frames
from report_native_joint_events import joint_events, aggregate as aggregate_joint
from report_native_tolerant_events import evaluate

POPCOUNT = np.array([int(i).bit_count() for i in range(256)], np.uint8)


def require_packed(packed, native_shape):
    n, h, w = native_shape
    require(all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                and v > 0 for v in native_shape), 'Native integer shape required')
    require(isinstance(packed, np.ndarray) and packed.dtype == np.uint8
            and packed.shape == (n, (h*w+7)//8), 'Complete original packed masks required')
    if h*w % 8:
        require(not (packed[:, -1] & ((255 << (h*w % 8)) & 255)).any(), 'Nonzero mask padding')


def load_target(record, reader):
    
    n, h, w = (record[k] for k in ('frame_count', 'height', 'width'))
    packed = np.zeros((n, (h*w+7)//8), np.uint8)
    fidelity = 'negative_no_effect'
    if record['role'] == 'bug':
        path = Path(record['effect_mask_manifest'])
        manifest = reader.read(path, record['effect_mask_manifest_sha256'])
        require(tuple(manifest[k] for k in ('frame_count', 'height', 'width')) == (n, h, w), 'GT native coordinates differ')
        fidelity = manifest['pixel_supervision_fidelity']
        require(isinstance(fidelity, str) and bool(fidelity), 'GT fidelity must be preserved')
        seen = set()
        for item in manifest['effect_masks']:
            t = item['frame_index']
            require(type(t) is int and 0 <= t < n and t not in seen, 'Repeated/out-of-range GT frame')
            seen.add(t); reader.bind(item['path'], item['sha256'])
            with Image.open(item['path']) as image: mask = np.asarray(image.convert('L')) > 0
            require(mask.shape == (h, w) and int(mask.sum()) == item['changed_pixel_count'] > 0, 'GT pixel content differs')
            packed[t] = np.packbits(mask.ravel(), bitorder='little')
        require(seen == set(manifest['nonzero_mask_frames'])
                and len(manifest['nonzero_mask_frames']) == len(seen), 'Sparse GT coverage differs')
    else:
        require(record['role'] in ('clean_or_fixed', 'hard_negative'), 'Unsupported source role')
    require_packed(packed, (n, h, w))
    require(packed.any(axis=1).astype(int).tolist() == frame_labels(record), 'Original pixel/event labels disagree')
    return packed, fidelity


def native_frame_metrics(predicted, target, native_shape):
    
    require_packed(predicted, native_shape); require_packed(target, native_shape)
    n, h, w = native_shape
    p = POPCOUNT[predicted].sum(axis=1, dtype=np.int64)
    g = POPCOUNT[target].sum(axis=1, dtype=np.int64)
    tp = POPCOUNT[predicted & target].sum(axis=1, dtype=np.int64)
    rows = []
    empty = dict(predicted_boundary_pixels=0, target_boundary_pixels=0,
                 matched_predicted_boundary_pixels=0, matched_target_boundary_pixels=0)
    for t in range(n):
        area, foreground, hit = int(g[t]), int(p[t]), int(tp[t]); fraction = area/(h*w)
        pixel = count_metrics(dict(tp=hit, fp=foreground-hit, fn=area-hit, tn=h*w-foreground-area+hit))
        row = dict(frame_index=t, target_positive_pixels=area, predicted_positive_pixels=foreground,
                   target_area_fraction=fraction,
                   target_area_bin='zero' if not area else 'le_1pct' if fraction<=.01 else '1_to_5pct' if fraction<=.05 else '5_to_35pct' if fraction<=.35 else 'gt_35pct',
                   pixels=pixel)
        if not area and not foreground:
            row['boundary_exact'] = boundary_metrics(empty)
            row['boundary_2px'] = boundary_metrics(empty)
        else:
            pm, gm = unpack_mask(predicted[t], h, w), unpack_mask(target[t], h, w)
            row['boundary_exact'] = boundary_metrics(boundary_counts(pm, gm, 0))
            row['boundary_2px'] = boundary_metrics(boundary_counts(pm, gm, 2))
        rows.append(row)
    return rows


def score_source(record, prediction, target, fidelity):
    
    shape = tuple(record[k] for k in ('frame_count', 'height', 'width'))
    require_original_replay(prediction, prediction)
    require(tuple(prediction['native_shape']) == shape, 'Prediction/GT original shape differs')
    require_packed(target, shape)
    truth = frame_labels(record)
    require(target.any(axis=1).astype(int).tolist() == truth, 'Frozen pixel/time labels disagree')
    frames = native_frame_metrics(prediction['packed_masks'], target, shape)
    row = dict(sample_id=record['sample_id'], frame_count=shape[0], fps=record['fps'],
               frame_targets=truth, frame_probabilities=np.asarray(prediction['frame_probabilities']).tolist(),
               source_time_s=np.asarray(prediction['source_time_s']).tolist(), all_source_frames_covered=True,
               role=record['role'], root_cause_family=record['root_cause_family'],
               camera_condition=record['camera_condition'], scene_id=record['scene_id'],
               video_sha256=record['video_sha256'], fidelity=fidelity)
    if record['role'] == 'hard_negative': row['hard_negative_category'] = record['hard_negative_category']
    validate_frames(frames, row)
    return dict(prediction=row, frames=frames, pixel_summary=summarize_frames(frames),
                joint_events=joint_events(row, frames, 3))


def aggregate_cases(cases):
    
    require(bool(cases), 'Empty metric group')
    rows = [c['prediction'] for c in cases]
    require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate original source')
    for case in cases: validate_frames(case['frames'], case['prediction'])
    y = np.concatenate([np.asarray(r['frame_targets'], bool) for r in rows])
    p = np.concatenate([np.asarray(r['frame_probabilities']) >= .5 for r in rows])
    tp, fp, fn, tn = (int((a & b).sum()) for a, b in ((y,p), (~y,p), (y,~p), (~y,~p)))
    video_y = np.asarray([any(r['frame_targets']) for r in rows], bool)
    video_p = np.asarray([any(v>=.5 for v in r['frame_probabilities']) for r in rows], bool)
    vt, vf, vn, vtn = (int((a & b).sum()) for a, b in ((video_y,video_p),(~video_y,video_p),(video_y,~video_p),(~video_y,~video_p)))
    flat = [dict(f, sample_id=c['prediction']['sample_id']) for c in cases for f in c['frames']]
    positive_cases = [c for c in cases if any(c['prediction']['frame_targets'])]
    def mean(values): return float(np.mean(values)) if values else None
    def frac(a, b): return a/b if b else None
    pixel_summary = summarize_frames(flat)
    pixel_summary['equal_positive_case_mean_frame_iou'] = mean([c['pixel_summary']['mean_positive_frame_iou'] for c in positive_cases])
    pixel_summary['equal_positive_case_mean_boundary2_f1'] = mean([c['pixel_summary']['boundary_2px']['mean_positive_frame_f1'] for c in positive_cases])
    joint = aggregate_joint([c['joint_events'] for c in cases])
    event = {str(t): evaluate(rows, t) for t in (0,3)}
    require(all(joint['temporal_'+k] == event['3'][k] for k in ('tp','fp','fn')), 'Joint scorer changed original temporal matches')
    normal = {role: [r for r in rows if r['role'] == role] for role in ('clean_or_fixed','hard_negative')}
    negative = {role: dict(videos=len(items), false_positive_videos=sum(any(p>=.5 for p in r['frame_probabilities']) for r in items))
                for role,items in normal.items()}
    return dict(videos=len(rows), source_frames=len(y), threshold=.5, tolerance_original_frames=3,
        video=dict(tp=vt, fp=vf, fn=vn, tn=vtn, accuracy=frac(vt+vtn,len(rows)),
                   f1=frac(2*vt,2*vt+vf+vn), recall=frac(vt,vt+vn), specificity=frac(vtn,vtn+vf)),
        frame=dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=frac(2*tp,2*tp+fp+fn),recall=frac(tp,tp+fn)),
        event={key:{k:v for k,v in value.items() if k!='records'} for key,value in event.items()},
        temporal_details=temporal_strata(rows), native_pixels=pixel_summary,
        strict_spatial_precision90_recall90=spatial_quality(flat), joint_time_space=joint,
        negatives=negative, boundary_distance_unit='original source pixels; Chebyshev2, not temporal tolerance',
        source_frame_alignment='strict same-frame pixel counts; +/-3 applies only to endpoint event matching',
        automatic_promotion=False)
