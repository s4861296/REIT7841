
import numpy as np
import cv2


def require_mask(value):
    value=np.asarray(value)
    if value.ndim!=2 or value.dtype!=bool or min(value.shape)<1:
        raise ValueError('Expected nonempty native 2D boolean mask, not probabilities/boxes')
    return value


def pixel_counts(predicted,target):
    predicted=require_mask(predicted);target=require_mask(target)
    if predicted.shape!=target.shape:raise ValueError('Native mask coordinate dimensions differ')
    return {'tp':int((predicted&target).sum()),'fp':int((predicted&~target).sum()),
            'fn':int((~predicted&target).sum()),'tn':int((~predicted&~target).sum())}


def count_metrics(counts):
    tp,fp,fn,tn=(int(counts[k]) for k in ('tp','fp','fn','tn'))
    if min(tp,fp,fn,tn)<0:raise ValueError('Negative confusion count')
    return {**counts,'iou':tp/(tp+fp+fn) if tp+fp+fn else None,
            'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'negative_pixel_fp_rate':fp/(fp+tn) if fp+tn else None}


def inner_boundary(mask):
    
    mask=require_mask(mask)
    cross=np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
    interior=cv2.erode(mask.astype(np.uint8),cross,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    return mask&~interior


def boundary_counts(predicted,target,radius=0):
    
    predicted=require_mask(predicted);target=require_mask(target)
    if predicted.shape!=target.shape or not isinstance(radius,int) or radius<0:
        raise ValueError('Invalid dimensions or boundary radius')
    p,t=inner_boundary(predicted),inner_boundary(target)
    if radius:
        kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
        near_t=cv2.dilate(t.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        near_p=cv2.dilate(p.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    else:near_t,near_p=t,p
    return {'predicted_boundary_pixels':int(p.sum()),'target_boundary_pixels':int(t.sum()),
            'matched_predicted_boundary_pixels':int((p&near_t).sum()),
            'matched_target_boundary_pixels':int((t&near_p).sum())}


def boundary_metrics(counts):
    p,t,mp,mt=(int(counts[k]) for k in ['predicted_boundary_pixels','target_boundary_pixels',
                                        'matched_predicted_boundary_pixels','matched_target_boundary_pixels'])
    if min(p,t,mp,mt)<0 or mp>p or mt>t:raise ValueError('Invalid boundary matching counts')
    precision=mp/p if p else None;recall=mt/t if t else None
    if not p and not t:f1=None
    elif not p or not t:f1=0.
    else:f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return {**counts,'precision':precision,'recall':recall,'f1':f1}


def frame_metrics(predicted,target,frame_index):
    counts=pixel_counts(predicted,target)
    area=counts['tp']+counts['fn'];fraction=area/target.size
    bin_name='zero' if not area else 'le_1pct' if fraction<=.01 else '1_to_5pct' if fraction<=.05 else '5_to_35pct' if fraction<=.35 else 'gt_35pct'
    return {'frame_index':int(frame_index),'target_positive_pixels':area,'predicted_positive_pixels':counts['tp']+counts['fp'],
            'target_area_fraction':fraction,'target_area_bin':bin_name,'pixels':count_metrics(counts),
            'boundary_exact':boundary_metrics(boundary_counts(predicted,target,0)),
            'boundary_2px':boundary_metrics(boundary_counts(predicted,target,2))}


def summarize_frames(rows):
    if not rows:raise ValueError('No pixel evaluations to summarize')
    positive=[r for r in rows if r['target_positive_pixels']>0]
    normal=[r for r in rows if r['target_positive_pixels']==0]
    def mean(values):return float(np.mean(values)) if values else None
    pixels=count_metrics({k:sum(r['pixels'][k] for r in rows) for k in ['tp','fp','fn','tn']})
    result={'frames':len(rows),'positive_frames':len(positive),'normal_frames':len(normal),
            'micro_native_pixel':pixels,
            'mean_positive_frame_iou':mean([r['pixels']['iou'] for r in positive]),
            'normal_frames_with_any_false_pixel':sum(r['predicted_positive_pixels']>0 for r in normal),
            'normal_mean_false_pixel_count':mean([r['predicted_positive_pixels'] for r in normal])}
    for metric in ['boundary_exact','boundary_2px']:
        result[metric]=boundary_metrics({k:sum(r[metric][k] for r in rows) for k in [
            'predicted_boundary_pixels','target_boundary_pixels','matched_predicted_boundary_pixels','matched_target_boundary_pixels']})
        result[metric]['mean_positive_frame_f1']=mean([r[metric]['f1'] for r in positive])
    return result


def pack_mask(mask):
    
    mask=require_mask(mask)
    return np.packbits(mask.ravel(),bitorder='little')


def unpack_mask(packed,height,width):
    packed=np.asarray(packed)
    size=height*width
    if packed.dtype!=np.uint8 or packed.ndim!=1 or min(height,width)<1 or len(packed)!=(size+7)//8:
        raise ValueError('Packed native mask size/type mismatch')
    return np.unpackbits(packed,bitorder='little',count=size).reshape(height,width).astype(bool)
