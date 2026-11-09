
from pathlib import Path
import json
import av
import numpy as np
from PIL import Image
import torch
import train_r3d18_flicker_detector as common
from r3d_native_pixel_refiner import native_inputs


def load_native_video(record,context,bind):
    path=Path(record['video_path']);bind(path,record['video_sha256'])
    n,h,w=record['frame_count'],record['height'],record['width']
    frames=[];times=[]
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_count=2
        if abs(float(stream.average_rate)-record['fps'])>1e-4:raise ValueError('Original FPS mismatch')
        for frame in container.decode(video=0):
            rgb=frame.to_ndarray(format='rgb24')
            if rgb.shape!=(h,w,3) or frame.time is None:raise ValueError('Native shape/timestamp mismatch')
            frames.append(rgb);times.append(float(frame.time))
    if len(frames)!=n or not np.allclose(np.diff(times),1/record['fps'],atol=1e-4):raise ValueError('Incomplete native sequence')
    rgb=np.stack(frames);del frames
    masks=np.zeros((n,h,w),bool);fidelity='negative_no_effect'
    if record['role']=='bug':
        mp=Path(record['effect_mask_manifest']);bind(mp,record['effect_mask_manifest_sha256']);manifest=json.loads(mp.read_text())
        if (manifest['frame_count'],manifest['height'],manifest['width'])!=(n,h,w):raise ValueError('GT dimensions differ')
        fidelity=manifest['pixel_supervision_fidelity'];seen=set()
        for item in manifest['effect_masks']:
            t=item['frame_index'];mp=Path(item['path']);bind(mp,item['sha256'])
            if t in seen or not 0<=t<n:raise ValueError('Invalid sparse GT indices')
            seen.add(t)
            with Image.open(mp) as image:mask=np.asarray(image.convert('L'))>0
            if mask.shape!=(h,w) or int(mask.sum())!=item['changed_pixel_count']:raise ValueError('GT pixel content mismatch')
            masks[t]=mask
        if seen!=set(manifest['nonzero_mask_frames']):raise ValueError('Sparse GT coverage mismatch')
    elif record['role'] not in ('clean_or_fixed','hard_negative'):
        raise ValueError('Unsupported sample role')
    targets=common.frame_targets(record).astype(bool)
    if not np.array_equal(masks.any((1,2)),targets):raise ValueError('Native pixel/event labels disagree')
    for key in ['sample_id','pair_id','role','frame_count','fps','camera_condition']:
        if context[key]!=record[key]:raise ValueError('Frozen R3D context identity mismatch: '+key)
    if context['frame_targets']!=targets.astype(int).tolist():raise ValueError('Frozen context GT differs')
    p=np.asarray(context['frame_probabilities'],np.float64)
    if p.shape!=(n,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid context scores')
    p=np.clip(p,1e-6,1-1e-6);z=np.log(p/(1-p)).astype(np.float32)
    bind(path,record['video_sha256'])
    return {'rgb':rgb,'masks':masks,'source_time_s':np.asarray(times,np.float64),'logits':z,
            'positive_pixels':int(masks.sum()),'negative_pixels':int(masks.size-masks.sum()),'fidelity':fidelity}


def feature_chunk(data,left,right,fps,device='cuda'):
    
    rgb,z=data['rgb'],data['logits'];n=len(rgb)
    if not 0<=left<right<=n or z.shape!=(n,):raise ValueError('Invalid source chunk')
    indices=np.arange(left,right)
    def tensor(ids):return torch.from_numpy(np.ascontiguousarray(rgb[ids].transpose(0,3,1,2))).to(device=device,dtype=torch.float32)/255.
    current=tensor(indices);previous=tensor(np.maximum(0,indices-1));following=tensor(np.minimum(n-1,indices+1))
    reference=(previous+following)/2
    if left==0:reference[0]=following[0]
    if right==n:reference[-1]=previous[-1]
    return native_inputs(current,previous,following,reference,torch.from_numpy(z[left:right]).to(device),
                         torch.from_numpy(indices>0).to(device),torch.from_numpy(indices<n-1).to(device),fps)
