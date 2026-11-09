
from __future__ import annotations
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import numpy as np

LAGS = (1,2,3,4,8,16)
CONTRAST_THRESHOLDS = (.03,.05,.10,.20)
FORBIDDEN = {'label','binary_label','gt','gt_label','ground_truth','events','symptom','symptoms','roi','clean_path'}

def require(ok,message):
    if not ok: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def label_free(value):
    if isinstance(value,dict):
        require(not (set(value)&FORBIDDEN),'Labels/GT regions forbidden')
        for item in value.values():label_free(item)
    elif isinstance(value,list):
        for item in value:label_free(item)

def validate_scope(row):
    label_free(row)
    indices,pts,times=row['source_frame_indices'],row['source_pts'],row['source_time_s']
    require(len(indices)==len(pts)==len(times)==row['frame_count']>0,'Complete native clock required')
    require(all(type(v) is int for v in indices+pts),'Integer native indices/PTS required')
    require(indices==list(range(indices[0],indices[0]+len(indices))),'One contiguous source interval required')
    require(all(b>a for a,b in zip(pts,pts[1:])),'Repeated/dropped/reordered PTS')
    base=Fraction(row['time_base_fraction'])
    require(base>0 and Fraction(row['fps_fraction'])>0,'Positive source clock required')
    require(np.isfinite(times).all() and np.allclose(times,[float(p*base) for p in pts],rtol=0,atol=1e-9),'PTS/time mismatch')
    sw,sh=row['source_width'],row['source_height'];crop=row.get('crop_xyxy')
    if crop is None:box=[0,0,sw,sh]
    else:
        require(row['sample_id']=='H13' and row.get('crop_is_user_requested_input_viewport_not_abnormal_gt_roi') is True,
                'Only frozen human-approved H13 viewport allowed')
        require(len(crop)==4 and all(type(v) is int for v in crop),'Integer viewport required');box=crop
    x0,y0,x1,y1=box
    require(0<=x0<x1<=sw and 0<=y0<y1<=sh and (x1-x0,y1-y0)==(row['width'],row['height']),'Viewport/native dimensions changed')
    require(min(row['height'],row['width'])>=8,'Native viewport too small for nonempty8x8 tiles')
    require(row['sha256']==row['video_sha256'],'Conflicting source SHA')
    return box

def tile_layout(height,width):
    require(min(height,width)>=8,'Every native tile must contain pixels')
    boxes=[(0,0,height,width)];groups=[1]
    for grid in (4,8):
        yy=np.arange(grid+1)*height//grid;xx=np.arange(grid+1)*width//grid
        for y in range(grid):
            for x in range(grid):boxes.append((int(yy[y]),int(xx[x]),int(yy[y+1]),int(xx[x+1])));groups.append(grid)
    area=np.asarray([(b-a)*(d-c)/(height*width) for a,c,b,d in boxes],np.float64)
    return boxes,area,np.asarray(groups,np.int64)

def rgb_luma(rgb):
    require(rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3,'Native RGB24 required')
    
    out=np.multiply(rgb[:,:,0],np.float32(.2126/255),dtype=np.float32)
    out+=np.multiply(rgb[:,:,1],np.float32(.7152/255),dtype=np.float32)
    out+=np.multiply(rgb[:,:,2],np.float32(.0722/255),dtype=np.float32)
    return out

def frame_statistics(luma,boxes):
    stats=np.empty((len(boxes),3),np.float64)
    for i,(y0,x0,y1,x1) in enumerate(boxes):
        block=luma[y0:y1,x0:x1]
        stats[i]=[np.mean(block,dtype=np.float64),np.count_nonzero(block<=np.float32(10/255))/block.size,
                  np.count_nonzero(block>=np.float32(245/255))/block.size]
    return stats

def blink_features(values,source_time_s,area_weights):
    
    x=np.asarray(values,np.float64);times=np.asarray(source_time_s,np.float64);weights=np.asarray(area_weights,np.float64)
    require(x.ndim==2 and x.shape[1]==81 and times.shape==(len(x),) and weights.shape==(81,),'Unpaired native statistic arrays')
    require(np.isfinite(x).all() and np.isfinite(times).all() and (np.diff(times)>0).all(),'Invalid native statistics/clock')
    require((weights>0).all() and all(np.isclose(weights[a:b].sum(),1) for a,b in ((0,1),(1,17),(17,81))),'Native area weights must sum1 per grid')
    n=len(x);strength=np.full((n,len(LAGS),81),np.nan,np.float64);polarity=np.full_like(strength,np.nan)
    valid=np.zeros((n,len(LAGS)),bool);span=np.full((n,len(LAGS),2),np.nan,np.float64)
    coverage=np.full((n,len(LAGS),2,len(CONTRAST_THRESHOLDS)),np.nan,np.float64)
    for j,lag in enumerate(LAGS):
        if n<=2*lag:continue
        before,center,after=x[:-2*lag],x[lag:-lag],x[2*lag:]
        amplitude=np.maximum(0,np.minimum(np.abs(center-before),np.abs(center-after))-np.abs(before-after))
        strength[lag:-lag,j]=amplitude;polarity[lag:-lag,j]=np.sign(center-(before+after)/2);valid[lag:-lag,j]=True
        span[lag:-lag,j,0]=times[lag:-lag]-times[:-2*lag];span[lag:-lag,j,1]=times[2*lag:]-times[lag:-lag]
        for g,(start,end) in enumerate(((1,17),(17,81))):
            for k,threshold in enumerate(CONTRAST_THRESHOLDS):
                coverage[lag:-lag,j,g,k]=(amplitude[:,start:end]>=threshold)@weights[start:end]
    any_valid=valid.any(axis=1)
    global_max=np.max(np.where(valid,strength[:,:,0],-np.inf),axis=1);global_max[~any_valid]=np.nan
    coverage_max=np.max(np.where(valid[:,:,None,None],coverage,-np.inf),axis=1);coverage_max[~any_valid]=np.nan
    return dict(blink_lags=np.asarray(LAGS,np.int64),blink_strength=strength,blink_polarity=polarity,
        blink_valid=valid,blink_elapsed_seconds=span,contrast_thresholds=np.asarray(CONTRAST_THRESHOLDS),
        blink_tile_coverage=coverage,global_blink_max=global_max,blink_tile_coverage_max_lag=coverage_max)

def collect_frames(row,frames,include_adjacent_difference=True):
    
    box=validate_scope(row);x0,y0,x1,y1=box
    boxes,weights,grids=tile_layout(row['height'],row['width']);n=row['frame_count']
    stats=np.empty((n,81,3),np.float64);diff=np.full(n,np.nan,np.float64);previous=None;count=0
    for pts,base,rgb in frames:
        require(count<n and pts==row['source_pts'][count] and Fraction(base)==Fraction(row['time_base_fraction']),
                'Dropped, extra, repeated or reordered selected source frame')
        require(rgb.shape==(row['source_height'],row['source_width'],3),'Original RGB geometry changed')
        luma=rgb_luma(rgb[y0:y1,x0:x1]);stats[count]=frame_statistics(luma,boxes)
        if include_adjacent_difference and previous is not None:
            delta=np.subtract(luma,previous);np.abs(delta,out=delta);diff[count]=np.mean(delta,dtype=np.float64)
        previous=luma if include_adjacent_difference else None;count+=1
    require(count==n,'Source ended before approved interval complete')
    arrays=dict(luminance_mean=stats[:,:,0],dark_fraction=stats[:,:,1],bright_fraction=stats[:,:,2],
        tile_area_weights=weights,tile_grid=grids,tile_boxes_yxyx=np.asarray(boxes,np.int64),
        source_frame_indices=np.asarray(row['source_frame_indices'],np.int64),source_pts=np.asarray(row['source_pts'],np.int64),
        source_time_s=np.asarray(row['source_time_s'],np.float64),source_time_base=np.asarray(row['time_base_fraction']),
        adjacent_luma_mean_absolute_difference=diff,adjacent_difference_valid=np.arange(n)>0 if include_adjacent_difference else np.zeros(n,bool))
    arrays.update(blink_features(arrays['luminance_mean'],arrays['source_time_s'],weights))
    return arrays

def extract_native(row,include_adjacent_difference=True):
    
    import av
    validate_scope(row);started=time.monotonic();source_hash=sha(row['video_path'])
    require(source_hash==row['sha256'],'Bound source video changed')
    preroll=0;decoded=0;pts=row['source_pts'];base=Fraction(row['time_base_fraction'])
    def frames():
        nonlocal preroll,decoded
        with av.open(row['video_path']) as container:
            require(len(container.streams.video)==1,'One source video stream required')
            stream=container.streams.video[0];stream.thread_count=2
            require(Fraction(stream.time_base)==base,'Original stream timebase changed')
            if pts[0]>0:container.seek(pts[0],stream=stream,backward=True,any_frame=False)
            for frame in container.decode(video=0):
                require(frame.pts is not None and Fraction(frame.time_base)==base,'Missing/changed original PTS')
                if frame.pts<pts[0]:preroll+=1;continue
                if frame.pts>pts[-1]:break
                require((frame.width,frame.height)==(row['source_width'],row['source_height']),'Original dimensions changed')
                yield frame.pts,frame.time_base,frame.to_ndarray(format='rgb24')
                decoded+=1
                if decoded==len(pts):break
    arrays=collect_frames(row,frames(),include_adjacent_difference)
    row_hash=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    receipt=dict(sample_id=row['sample_id'],source_video_sha256=source_hash,input_row_canonical_sha256=row_hash,
        frame_count=decoded,source_dimensions=[row['source_height'],row['source_width']],
        input_dimensions=[row['height'],row['width']],crop_xyxy=row.get('crop_xyxy'),
        source_pts_sha256=hashlib.sha256(arrays['source_pts'].tobytes()).hexdigest(),
        source_time_s_sha256=hashlib.sha256(arrays['source_time_s'].tobytes()).hexdigest(),
        extractor_sha256=sha(__file__),pyav_version=av.__version__,numpy_version=np.__version__,
        elapsed_seconds=time.monotonic()-started,codec_preroll_discarded_frames=preroll,decode_threads=2,
        source_resize=False,frame_dropping=False,FPS_resampling=False,RGB_scratch_written=False,GPU=False,
        labels_used=False,GT_ROI_used=False,model_training=False,
        luminance_definition="Y'=(.2126R+.7152G+.0722B)/255 on original gamma-coded RGB24; float32 per pixel,float64 means",
        dark_cutoff=10/255,bright_cutoff=245/255,tile_layout='global column0;4x4 row-major1:17;8x8 row-major17:81; integer native-pixel boundaries',
        blink_definition='max(0,min(abs(center-before),abs(center-after))-abs(before-after))',
        unknown_edges='NaN strength/coverage, valid=False; never count as observed normal',
        caveat='Appearance-return evidence may be normal lighting/motion/UI; not proof of a VR bug or improved localization')
    return arrays,receipt
