
from pathlib import Path
from fractions import Fraction
import json, hashlib, os, sys, subprocess, tempfile, shutil, time, fcntl
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LEGACY=ROOT/'mef_vr/legacy'
sys.path.insert(0,str(LEGACY))

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def status(out,**v):
    save(out/'status.json',dict(time=time.time(),**v))
    print(json.dumps(v,ensure_ascii=False),flush=True)
def config():
    p=read(ROOT/'configs/detector.json')
    expected=dict(full_threshold=.5,context_threshold=.9,rife_confirmation_error=.01,
        rife_rescue_error=.04,rife_ratio=.25,
        rife_denominator='mean over pixels of min(previous RGB L1,next RGB L1)',
        rife_epsilon=1/255,neighbour_radius_frames=3,endpoint_difference=1/255,
        dark_lags=[1,2,3,4,8,16],dark_amplitude=.05,dark_area=.5,
        weak_full=.25,weak_context=.5,inference='offline single observed video')
    if p['complete_detector']!=expected:
        raise ValueError('Configuration differs from the archived complete detector; changing a description is not an algorithm update')
    p['models']={k:str(ROOT/v) for k,v in p['models'].items()}
    p['heads']={k:dict(v,path=str(ROOT/v['path'])) for k,v in p['heads'].items()}
    p['model_bindings']={str(ROOT/k):v for k,v in p['model_bindings'].items()}
    p['normalization_path']=str(ROOT/p['normalization_path'])
    p['rife_asset_root']=str(ROOT/p['rife_asset_root'])
    return p

def verify_models(p):
    for path,digest in p['model_bindings'].items():
        if sha(path)!=digest:raise ValueError('Model checksum mismatch: '+path)
    for path,digest in p['source_bindings'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Detector implementation changed: '+path)

def source_scope(video):
    import av
    video=Path(video).resolve(strict=True);digest=sha(video)
    pts=[];times=[];shape=None
    with av.open(str(video)) as c:
        if len(c.streams.video)!=1:raise ValueError('Exactly one video stream is required')
        stream=c.streams.video[0];stream.thread_count=2;base=Fraction(stream.time_base)
        fps=Fraction(stream.average_rate or stream.guessed_rate)
        for frame in c.decode(video=0):
            if frame.pts is None:raise ValueError('Missing presentation timestamp')
            if Fraction(frame.time_base)!=base:raise ValueError('Changing source time base')
            if shape is None:shape=(frame.height,frame.width)
            if shape!=(frame.height,frame.width):raise ValueError('Changing image dimensions')
            pts.append(int(frame.pts));times.append(float(frame.pts*base))
    if len(pts)<2 or any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('Need at least two monotonically timestamped frames')
    h,w=shape
    return dict(sample_id='video_'+digest[:20],video_path=str(video),sha256=digest,video_sha256=digest,
        frame_count=len(pts),source_width=w,source_height=h,width=w,height=h,crop_xyxy=None,
        source_frame_indices=list(range(len(pts))),source_pts=pts,source_time_s=times,
        time_base_fraction=str(base),fps_fraction=str(fps))

def resolve_scope(row):
    row=dict(row);p=Path(row['video_path']);row['video_path']=str(p if p.is_absolute() else ROOT/p)
    if sha(row['video_path'])!=row['video_sha256']:raise ValueError('Video checksum mismatch')
    row['sha256']=row['video_sha256']
    return row

def source_frames(row):
    import av
    pts=row['source_pts'];base=Fraction(row['time_base_fraction']);count=0
    with av.open(row['video_path']) as c:
        if len(c.streams.video)!=1:raise ValueError('Exactly one video stream is required')
        s=c.streams.video[0];s.thread_count=2
        if Fraction(s.time_base)!=base:raise ValueError('Source time base differs')
        if pts[0]>0:c.seek(pts[0],stream=s,backward=True,any_frame=False)
        for f in c.decode(video=0):
            if f.pts is None:raise ValueError('Missing timestamp')
            if f.pts<pts[0]:continue
            if f.pts>pts[-1]:break
            if count>=len(pts) or int(f.pts)!=pts[count]:raise ValueError('Source clock differs')
            if (f.height,f.width)!=(row['source_height'],row['source_width']):raise ValueError('Source resized')
            yield int(f.pts),base,f.to_ndarray(format='rgb24')
            count+=1
            if count==len(pts):break
    if count!=len(pts):raise ValueError('Source ended before selected scope')

def gpu_worker(out,stage,ablations=False):
    p=config();verify_models(p);row=read(out/'source.json')
    lock_path=Path(os.environ.get('MEF_VR_GPU_LOCK','/tmp/vrflicker_real_gt_three_models_20261014_gpu.lock'))
    with lock_path.open('a+') as lock:
        status(out,stage='waiting_for_gpu',worker=stage)
        fcntl.flock(lock,fcntl.LOCK_EX)
        if stage=='rife':
            import torch,native_sequence
            from rife_adapter import RIFEAdapter
            torch.set_num_threads(p['threads']);torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
            native_sequence.SCOPE_HELPER=LEGACY/'native_photometric.py'
            model=RIFEAdapter(device='cuda:0',asset_root=p['rife_asset_root'])
            arrays,receipt=native_sequence.extract_native(row,model.estimate,
                progress=lambda v:status(out,stage='RIFE',progress=v))
            np.savez_compressed(out/'native_interpolation.npz',**arrays)
            save(out/'rife_provenance.json',dict(extraction=receipt,model=model.provenance()))
            return
        import torch
        import infer_flicker_retained_parent_cuda as base
        import run_flicker_multimodel_observed_inference as components
        import run_native_execution as native
        base.configure_observation_cuda(p)
        profile=row.get('memory_profile')
        if profile is None and row['height']*row['width']>=2560*1440:
            cells=((row['height']+7)//8)*((row['width']+7)//8)
            profile=dict(raft='inplace' if cells*cells*4<=12.5*1024**3 else 'recompute',
                         refiner_bounded=row['height']*row['width']>=3840*2160)
        if profile:
            import importlib.util
            def load_adapter(name,path):
                spec=importlib.util.spec_from_file_location(name,path)
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
            directory=ROOT/'mef_vr/memory_adapter'
            adapter=load_adapter('mef_raft_adapter',directory/'raft_memory_adapter.py')
            receipt=dict(raft=adapter.install_raft_adapter(LEGACY,mode=profile['raft'],
                parity_receipt=directory/'gpu_parity_receipt_v1.json',max_chunk_bytes=256*1024**2))
            if profile['refiner_bounded']:
                directory=ROOT/'mef_vr/native_c_adapter'
                adapter=load_adapter('mef_native_adapter',directory/'bounded_features.py')
                adapter.CHECKPOINT=Path(p['models']['native_C'])
                receipt['refiner']=adapter.install(LEGACY,parity_receipt=directory/'gpu_parity_receipt.json')
            save(out/'memory_execution_receipt.json',receipt)
        scratch=out/'work';scratch.mkdir(exist_ok=False)
        n,h,w=row['frame_count'],row['height'],row['width']
        if shutil.disk_usage(out).free < n*h*w*3 + 40*1024**3:raise OSError('Keep 40 GiB free beyond native RGB scratch')
        rgb=np.lib.format.open_memmap(scratch/'rgb.npy',mode='w+',dtype=np.uint8,shape=(n,h,w,3))
        box=row.get('crop_xyxy') or [0,0,row['source_width'],row['source_height']]
        x0,y0,x1,y1=box
        for t,(_,_,frame) in enumerate(source_frames(row)):rgb[t]=frame[y0:y1,x0:x1]
        rgb.flush()
        obs=dict(row,fps=float(Fraction(row['fps_fraction'])))
        update=lambda **v:status(out,stage='primary',**v)
        scores=np.asarray(components.upstream(scratch,p,rgb,update)['frame_probabilities'],np.float64)
        packed,maxima=native.native_mask(scratch,p,rgb,obs,scores,update,base)
        pooled=components.motion(scratch,p,rgb,update)
        features=components.features(scratch,p,rgb,scores,packed,pooled,update)
        if read(scratch/'features.json')['feature_names']!=p['feature_names']:raise ValueError('51-channel schema mismatch')
        base.configure_temporal_cuda()
        models={k:v.cuda() for k,v in base.load_heads_cpu(p).items()}
        with np.load(p['normalization_path'],allow_pickle=False) as z:mean,std=z['mean'].copy(),z['std'].copy()
        prediction=base.predict_retained_parent(models['full'],models['frame_context'],features,mean,std,
            device='cuda:0',source_pts=row['source_pts'],source_time_s=row['source_time_s'],time_base_fraction=row['time_base_fraction'])
        arrays={kind+'_'+key:value for kind in ('full','frame_context') for key,value in prediction[kind].items()}
        np.savez_compressed(out/'temporal_predictions.npz',**arrays,
            source_pts=np.asarray(row['source_pts'],np.int64),source_frame_indices=np.asarray(row['source_frame_indices'],np.int64),
            source_time_s=np.asarray(row['source_time_s'],np.float64))
        np.savez_compressed(out/'region_candidates.npz',packed_masks=packed,native_shape=np.asarray((n,h,w)),
            source_pts=np.asarray(row['source_pts']),frame_probabilities=maxima)
        if ablations:
            from intervention import ARMS,predict_heads
            for arm in ARMS:
                if arm=='full':continue
                status(out,stage='feature_intervention',arm=arm)
                values=predict_heads(models,features,mean,std,arm,device='cuda:0')
                np.savez_compressed(out/f'heads_{arm}.npz',**{k+'_'+name:v for k,a in values.items() for name,v in a.items()})
        save(out/'primary_receipt.json',dict(source_video_sha256=row['video_sha256'],all_frames=n,
            feature_names=p['feature_names'],stage_precision=p['stage_precision'],paired_clean_input=False,
            region_output='Raw Refiner candidate pixels, not final-system pixel ground truth'))

def observation_from_files(row,files):
    t=npz(files['temporal_predictions.npz'])
    return dict(full=t['full_frame_probabilities'],context=t['frame_context_frame_probabilities'],
        photo=npz(files['native_photometric.npz']),interp=npz(files['native_interpolation.npz']),
        hist=npz(files['native_histogram_transport.npz']),
        clock={k:row[k] for k in ('source_frame_indices','source_pts','source_time_s','time_base_fraction')})

def decide_and_save(row,out,ablations=False):
    import decision_api
    names=('temporal_predictions.npz','native_photometric.npz','native_interpolation.npz','native_histogram_transport.npz')
    obs=observation_from_files(row,{n:out/n for n in names});answer=decision_api.predict(obs)
    result=dict(sample_id=row['sample_id'],status='completed',predicted_label=answer['video_prediction'],
        has_flicker=bool(answer['video_prediction']),source_video_sha256=row['video_sha256'],
        frame_count=row['frame_count'],fps_fraction=row['fps_fraction'],resolution=[row['height'],row['width']],
        detector='MEF-VR complete detector',counts=answer['counts'],factors=answer['video_factors'],
        configuration_sha256=sha(ROOT/'configs/detector.json'),
        temporal_output='Per-frame model scores and candidate support; no new localization validation claimed',
        region_output='Raw native Refiner candidate mask, separate from binary video output')
    save(out/'prediction.json',result)
    np.savez_compressed(out/'frame_evidence.npz',**answer['candidate_evidence_flags'],
        full_score=obs['full'],context_score=obs['context'],source_pts=np.asarray(row['source_pts']),source_time_s=np.asarray(row['source_time_s']))
    if ablations:
        import ablation_rules
        variants=ablation_rules.predict_all(obs)
        for arm in ('no_dino','no_raft','no_r3d_lineage','no_c_mask'):
            a=npz(out/f'heads_{arm}.npz')
            variants['input_'+arm]=decision_api.predict(dict(obs,full=a['full_frame_probabilities'],context=a['frame_context_frame_probabilities']))['video_prediction']
        save(out/'interventions.json',variants)
    return result

def detect(row,out,*,ablations=False,keep_features=False):
    row=resolve_scope(row);out=Path(out).resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory; existing work is preserved: '+str(out))
    out.mkdir(parents=True)
    clock={k:row[k] for k in ('source_frame_indices','source_pts','source_time_s')}
    save(out/'input_clock.json',clock)
    row['input_clock_path']=str(out/'input_clock.json');row['input_clock_sha256']=sha(out/'input_clock.json')
    save(out/'source.json',row)
    save(out/'execution_protocol.json',dict(config_sha256=sha(ROOT/'configs/detector.json'),all_modules=True,labels_in_forward=False))
    began=time.monotonic()
    try:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        for stage in ('primary','rife'):
            command=[sys.executable,str(ROOT/'run.py'),'--worker',stage,'--output',str(out)]
            if ablations:command.append('--ablations')
            subprocess.run(command,env=env,check=True)
        import native_photometric,histogram_transport
        histogram_transport.SCOPE_HELPER=LEGACY/'native_photometric.py'
        status(out,stage='photometric_and_adjacent_evidence')
        photo=native_photometric.collect_frames(row,source_frames(row))
        hist=histogram_transport.collect_frames(row,source_frames(row))
        np.savez_compressed(out/'native_photometric.npz',**photo)
        np.savez_compressed(out/'native_histogram_transport.npz',**hist)
        result=decide_and_save(row,out,ablations)
        if not keep_features:
            scratch=out/'work'
            if scratch.is_dir() and not scratch.is_symlink() and scratch.parent==out:shutil.rmtree(scratch)
        else:
            
            for f in (out/'work').iterdir():
                if f.name not in ('features.npy','features.json') and f.is_file():f.unlink()
        save(out/'completed.json',dict(completed=True,seconds=time.monotonic()-began,
            source_video_sha256=row['video_sha256'],prediction_sha256=sha(out/'prediction.json')))
        status(out,stage='completed',seconds=round(time.monotonic()-began,2),has_flicker=result['has_flicker'])
        return result
    except BaseException as e:
        save(out/'failure.json',dict(status='failed',sample_id=row['sample_id'],predicted_label=None,error=repr(e)))
        raise
