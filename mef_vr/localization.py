
from pathlib import Path
import hashlib
import numpy as np
from . import runtime as rt
from .evaluation import count_metrics

def score(batch,output):
    from r3d_native_pixel_metrics import unpack_mask,frame_metrics,summarize_frames
    from report_native_tolerant_events import evaluate
    batch=Path(batch);output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh metric directory')
    job=rt.read(batch/'job.json');split=job['split']
    if split not in ('train','development'):raise ValueError('Real-97 has no equally complete native localization labels')
    labels={r['sample_id']:r for r in rt.read(rt.ROOT/f'data/labels/{split}.json')}
    sources={r['sample_id']:r for r in rt.read(rt.ROOT/f'data/manifests/{split}.json')}
    output.mkdir(parents=True);pixel=[];timing={'full':[],'frame_context':[]};failures=[]
    for sid in job['ids']:
        folder=batch/'cases'/hashlib.sha256(sid.encode()).hexdigest()[:20]
        if not (folder/'completed.json').exists():failures.append(sid);continue
        row=labels[sid];target=rt.ROOT/row['targets']
        if rt.sha(target)!=row['target_sha256']:raise ValueError('Target checksum differs')
        y=rt.npz(target);pred=rt.npz(folder/'region_candidates.npz');scores=rt.npz(folder/'temporal_predictions.npz')
        n,h,w=map(int,y['native_shape'])
        if list(pred['native_shape'])!=[n,h,w] or not np.array_equal(pred['source_pts'],sources[sid]['source_pts']):
            raise ValueError('Native target/prediction clock or size differs')
        frames=[frame_metrics(unpack_mask(pred['packed_masks'][t],h,w),unpack_mask(y['packed_masks'][t],h,w),t) for t in range(n)]
        pixel.append(dict(sample_id=sid,fidelity=row['pixel_supervision_fidelity'],**summarize_frames(frames)))
        rt.save(output/(hashlib.sha256(sid.encode()).hexdigest()[:20]+'.json'),frames)
        for head in timing:
            timing[head].append(dict(sample_id=sid,frame_count=n,fps=float(__import__('fractions').Fraction(sources[sid]['fps_fraction'])),
                frame_targets=y['frame'].astype(int).tolist(),frame_probabilities=scores[head+'_frame_probabilities'].tolist()))
    temporal={}
    for head,rows in timing.items():
        counts=dict(tp=0,tn=0,fp=0,fn=0);ious=[]
        for row in rows:
            y=np.array(row['frame_targets'],bool);p=np.array(row['frame_probabilities'])>=.5
            for k,v in dict(tp=(y&p).sum(),tn=(~y&~p).sum(),fp=(~y&p).sum(),fn=(y&~p).sum()).items():counts[k]+=int(v)
            if y.any():ious.append(float((y&p).sum()/(y|p).sum()))
        temporal[head]=dict(frame=count_metrics(counts),events=evaluate(rows,3) if rows else None,
            mean_positive_video_temporal_iou=float(np.mean(ious)) if ious else None)
    rt.save(output/'localization_results.json',dict(split=split,requested=len(job['ids']),scored=len(pixel),failed=failures,
        spatial_scope='Raw Refiner at 0.5, ungated native pixels; NOT complete-system validated localization',
        temporal_scope='Each fusion head separately at 0.5; NOT final complete detector candidate logic',
        pixel=pixel,temporal=temporal))
    print(dict(localization_videos=len(pixel),failed=len(failures),output=str(output)))
    if failures:raise RuntimeError('Some localization inputs were not completed')
