
import math
import numpy as np
import torch
from r3d_native_pixel_data import feature_chunk
from r3d_native_pixel_refiner import temporal_chunks


def collect_native_chunks(chunks, native_shape, source_time_s):
    
    if len(native_shape)!=3 or any(type(v) is not int or v<1 for v in native_shape):
        raise ValueError('Positive original T,H,W shape required')
    n,h,w=native_shape;times=np.asarray(source_time_s)
    if (times.shape!=(n,) or not np.issubdtype(times.dtype,np.floating)
            or not np.isfinite(times).all() or (np.diff(times)<=0).any()):
        raise ValueError('Complete increasing original source clock required')
    masks=np.empty((n,(h*w+7)//8),np.uint8);scores=np.empty(n,np.float64);emitted=0
    for chunk in chunks:
        first,last=chunk['start_frame'],chunk['end_frame_exclusive']
        p=chunk['pixel_probabilities']
        if (type(first) is not int or type(last) is not int or first!=emitted
                or not first<last<=n or not isinstance(p,np.ndarray)
                or p.shape!=(last-first,h,w) or not np.issubdtype(p.dtype,np.floating)
                or not np.isfinite(p).all() or (p<0).any() or (p>1).any()
                or not np.array_equal(chunk['source_time_s'],times[first:last])):
            raise ValueError('Missing/repeated/invalid native probability or source PTS')
        masks[first:last]=np.packbits((p>=.5).reshape(last-first,-1),axis=1,bitorder='little')
        scores[first:last]=p.max((1,2));emitted=last
    if emitted!=n:raise ValueError('Incomplete original-frame predictions')
    return masks,scores


def predict_zero_motion_chunks(model,rgb,upstream_probabilities,fps,source_time_s,*,device='cuda',return_logits=False):
    
    if type(return_logits) is not bool:raise ValueError('Explicit boolean logit-output choice required')
    if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=4 or rgb.shape[-1]!=3
            or min(rgb.shape[:3])<1 or any(m.training for m in model.modules())):raise ValueError('Complete native RGB/eval model required')
    n,h,w,_=rgb.shape;p=np.asarray(upstream_probabilities,np.float64);times=np.asarray(source_time_s)
    if p.shape!=(n,) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid same-video upstream probabilities')
    if times.shape!=(n,) or not np.issubdtype(times.dtype,np.floating) or not np.isfinite(times).all() or (np.diff(times)<=0).any():raise ValueError('Original PTS required')
    if not math.isfinite(float(fps)) or fps<=0:raise ValueError('Invalid original FPS')
    p=np.clip(p,1e-6,1-1e-6)
    observed={'rgb':rgb,'logits':np.log(p/(1-p)).astype(np.float32)}
    emitted=0
    for left,right,start,end in temporal_chunks(n,8,4):
        first,last=left+start,left+end
        if first!=emitted:raise RuntimeError('Missing/repeated source frames')
        with torch.inference_mode(),torch.autocast(device_type=torch.device(device).type,enabled=False):
            features=feature_chunk(observed,left,right,fps,device=device)
            cues=features.new_zeros((right-left,4,h,w))
            selected_logits=model(features[None],cues[None])[:,start:end]
            probability=selected_logits.sigmoid()[0].cpu().numpy()
            raw_logits=selected_logits[0].cpu().numpy() if return_logits else None
        if probability.shape!=(last-first,h,w) or not np.isfinite(probability).all():raise ValueError('Invalid native prediction')
        if return_logits and (raw_logits.dtype!=np.float32 or raw_logits.shape!=probability.shape or not np.isfinite(raw_logits).all()):
            raise ValueError('Finite actual FP32 native logits required')
        emitted=last;del features,cues,selected_logits
        result=dict(start_frame=first,end_frame_exclusive=last,pixel_probabilities=probability,source_time_s=times[first:last].copy())
        if return_logits:result['pixel_logits']=raw_logits
        yield result
    if emitted!=n:raise RuntimeError('Incomplete native output')
