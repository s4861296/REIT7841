
import math
import numpy as np
import torch
import torch.nn.functional as F
from r3d_native_pixel_data import feature_chunk
from r3d_native_pixel_refiner import temporal_chunks


def core_logits_cpu(model, features, cues, start, end):
    if model.use_native_skips is not False or features.dtype != torch.float32:
        raise ValueError('Only frozen C without native skips, FP32 supported')
    b,t,c,h,w = features.shape
    if b != 1 or c != 22 or cues.shape != (b,t,4,h,w):
        raise ValueError('Original single-video 22+4 channels required')
    if not (0 <= start < end <= t) or any(m.training for m in model.modules()):
        raise ValueError('Valid original core and eval model required')
    
    fine = model.stem(features.reshape(b*t,c,h,w))
    fine = fine + model.motion_projection(cues.reshape(b*t,4,h,w))
    mid = model.down2(fine)
    coarse = model.down4(mid)
    ch,cw = coarse.shape[-2:]
    sequence = coarse.reshape(b,t,32,ch,cw).permute(0,2,1,3,4)
    sequence = sequence + model.temporal(sequence)
    coarse = sequence.permute(0,2,1,3,4).reshape(b*t,32,ch,cw)
    mh,mw = mid.shape[-2:]
    
    del fine, mid, sequence
    output = np.empty((end-start,h,w), dtype=np.float32)
    for i in range(start,end):
        up = F.interpolate(coarse[i:i+1],size=(mh,mw),mode='bilinear',align_corners=False)
        up = model.up2(torch.cat((up,up.new_zeros((1,16,mh,mw))),dim=1))
        up = F.interpolate(up,size=(h,w),mode='bilinear',align_corners=False)
        up = model.up1(torch.cat((up,up.new_zeros((1,8,h,w))),dim=1))
        logits = model.classifier(up)
        output[i-start] = logits[0,0].cpu().numpy()
        del up, logits
    return output


def predict_native_core_chunks(model,rgb,upstream_probabilities,fps,source_time_s,
                               *,device='cuda',return_logits=False):
    if type(return_logits) is not bool or torch.device(device).type != 'cuda':
        raise ValueError('Explicit boolean return_logits and CUDA required; no fallback')
    if (not isinstance(rgb,np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 4
            or rgb.shape[-1] != 3 or min(rgb.shape[:3]) < 1
            or any(m.training for m in model.modules())):
        raise ValueError('Complete native RGB and eval model required')
    n,h,w,_ = rgb.shape
    p=np.asarray(upstream_probabilities,np.float64);times=np.asarray(source_time_s)
    if p.shape != (n,) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('Invalid same-video upstream probabilities')
    if (times.shape != (n,) or not np.issubdtype(times.dtype,np.floating)
            or not np.isfinite(times).all() or (np.diff(times)<=0).any()
            or not math.isfinite(float(fps)) or fps<=0):
        raise ValueError('Original FPS and complete increasing original PTS required')
    p=np.clip(p,1e-6,1-1e-6)
    observed={'rgb':rgb,'logits':np.log(p/(1-p)).astype(np.float32)}
    emitted=0
    for left,right,start,end in temporal_chunks(n,8,4):
        first,last=left+start,left+end
        if first != emitted:raise RuntimeError('Missing/repeated frames')
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            features=feature_chunk(observed,left,right,fps,device=device)
            cues=features.new_zeros((right-left,4,h,w))
            logits=core_logits_cpu(model,features[None],cues[None],start,end)
            
            probability=torch.from_numpy(logits).to(device).sigmoid().cpu().numpy()
            del features,cues
        if not np.isfinite(logits).all() or probability.shape != (last-first,h,w):
            raise ValueError('Invalid native FP32 prediction')
        result=dict(start_frame=first,end_frame_exclusive=last,
                    pixel_probabilities=probability,source_time_s=times[first:last].copy(),
                    execution_context=dict(left=left,right=right,core_start=start,core_end=end))
        if return_logits:result['pixel_logits']=logits
        emitted=last
        yield result
    if emitted != n:raise RuntimeError('Incomplete original output')
