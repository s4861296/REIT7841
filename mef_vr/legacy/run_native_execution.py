
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

os.environ['PYTHONDONTWRITEBYTECODE']='1'
HERE=Path(__file__).resolve().parent
REPO=Path('/mnt/c/Users/95774/Desktop/11111/vr-flicker-evidence-mining')
LOCK=Path('/mnt/e/VRFlickerLab/external_benchmarks/preparation_20261014_v1/gpu_evaluation.lock')
VERSION='retained600_oldC_core_only_native_decode_execution_v3'
NATIVE_DEPENDENCIES=('r3d_native_motion_cues.py','r3d_native_pixel_refiner.py',
    'r3d_native_pixel_data.py','r3d_native_restoration_aux.py','r3d_native_zero_motion_inference.py')


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def publish(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def prepare(video,out):
    
    parity=HERE/'cuda720p_parity_v1'
    report=read(parity/'report.json');contract=read(parity/'contract.json')
    if not (report['passed'] and report['completed'] and report['binary_flips']==0
            and report['total_frames']==120 and report['contract_sha256']==sha(parity/'contract.json')
            and contract['adapter_sha256']==sha(HERE/'native_core_decoder.py')):
        raise ValueError('Current adapter lacks bound passing CUDA parity')
    out=out.resolve();out.parent.mkdir(exist_ok=True,parents=True)
    if out.exists():raise ValueError('Fresh v3 output required; preserve all failed attempts')
    command=[sys.executable,str(REPO/'tools/infer_flicker_retained_parent_cuda.py'),
        '--action','prepare','--video',str(video.resolve()),'--output',str(out)]
    log=out.parent/(out.name+'.base_prepare.log')
    with log.open('x') as stream:
        checked=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),timeout=300)
    if checked.returncode:raise RuntimeError('Original preparation failed: '+str(log))
    p=read(out/'protocol.json')
    for n in NATIVE_DEPENDENCIES:
        if p['code_sha256'][n]!=contract['old_code_sha256'][n]:raise ValueError('Parity code differs: '+n)
    if p['model_bindings'][p['models']['native_C']]!=contract['checkpoint_sha256']:
        raise ValueError('Parity checkpoint differs')
    execution=out/'execution';execution.mkdir()
    names=('run_native_execution.py','native_core_decoder.py','run_cuda_parity.py')
    for n in names:shutil.copy2(HERE/n,execution/n)
    for n in ('report.json','contract.json'):shutil.copy2(parity/n,execution/('cuda_parity_'+n))
    spec=dict(version=VERSION,base_protocol_sha256=sha(out/'protocol.json'),
        base_prepared_sha256=sha(out/'prepared.json'),code_sha256={n:sha(execution/n) for n in names},
        code_provenance={n:dict(source_path=str(HERE/n),source_sha256=sha(HERE/n)) for n in names},
        parity_report_sha256=sha(execution/'cuda_parity_report.json'),
        parity_contract_sha256=sha(execution/'cuda_parity_contract.json'),
        changed_execution='Only post-temporal independent 2D native decoding; one useful core frame at a time',
        outer_core=8,outer_halo=4,maximum_encoder_temporal_context=16,native_decode_microbatch=1,
        model_parameter_changes=False,quantization_changes=False,original_model_selection=p['selection'],
        source_resize=False,frame_subsampling=False,fps_resampling=False,spatial_tiling=False,
        oldC_FP32=True,threshold=.5,original_other_stages_unchanged=True,shared_gpu_lock=str(LOCK),
        cuda_memory_fraction=.65,run_budget_seconds=1800,labels_used=False,paired_clean_used=False,
        numerical_equivalence_scope='Actual120-frame720pTrainRGB, fixed engineering upstream .1-.9; bitwise old/new; not all possible inputs proof')
    publish(out/'execution_protocol.json',spec)
    publish(out/'execution_prepared.json',dict(completed=True,
        execution_protocol_sha256=sha(out/'execution_protocol.json'),original_preparation_returncode=0,
        passing_cuda_regression=True))
    print(json.dumps(dict(prepared=str(out),entry=str(execution/'run_native_execution.py'),
                         execution_protocol_sha256=sha(out/'execution_protocol.json'))),flush=True)


def validate(out,base):
    if HERE!=out/'execution':raise ValueError('Run only the frozen explicit execution entry')
    p,source=base.validate_prepared(out)
    e=read(out/'execution_protocol.json');marker=read(out/'execution_prepared.json')
    if not (e['version']==VERSION and marker['completed']
            and marker['execution_protocol_sha256']==sha(out/'execution_protocol.json')
            and e['base_protocol_sha256']==sha(out/'protocol.json')
            and e['base_prepared_sha256']==sha(out/'prepared.json')
            and e['outer_core']==8 and e['outer_halo']==4 and e['native_decode_microbatch']==1
            and e['cuda_memory_fraction']==p['cuda_memory_fraction']==.65
            and e['threshold']==p['threshold']==.5):raise ValueError('Execution contract changed')
    base.verify({str(HERE/n):s for n,s in e['code_sha256'].items()})
    base.verify({str(HERE/'cuda_parity_report.json'):e['parity_report_sha256'],
                 str(HERE/'cuda_parity_contract.json'):e['parity_contract_sha256']})
    c=read(HERE/'cuda_parity_contract.json');r=read(HERE/'cuda_parity_report.json')
    if not (r['passed'] and r['binary_flips']==0 and c['adapter_sha256']==sha(HERE/'native_core_decoder.py')
            and r['contract_sha256']==sha(HERE/'cuda_parity_contract.json')
            and c['checkpoint_sha256']==p['model_bindings'][p['models']['native_C']]):
        raise ValueError('Numerical parity not bound to active adapter/checkpoint')
    for n in NATIVE_DEPENDENCIES:
        if p['code_sha256'][n]!=c['old_code_sha256'][n]:raise ValueError('Native model source changed')
    return p,source,e


def native_mask(folder,p,rgb,obs,scores,update,base):
    import numpy as np
    import torch
    from r3d_native_motion_cues import NativeMotionCueRefiner
    from r3d_native_zero_motion_inference import collect_native_chunks
    from native_core_decoder import predict_native_core_chunks
    if (folder/'native_C.npz').exists():raise ValueError('No partial-stage reuse')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    m=NativeMotionCueRefiner(False).cuda().eval().requires_grad_(False)
    c=torch.load(p['models']['native_C'],map_location='cpu',weights_only=True)
    if c['arm']!='C_zero_motion' or c['step']!=300:raise ValueError('Wrong C checkpoint')
    m.load_state_dict(c['model'],strict=True);del c
    times=np.asarray(obs['source_time_s'],np.float64);contexts=[]
    def chunks():
        for chunk in predict_native_core_chunks(m,rgb,scores,obs['fps'],times,device='cuda'):
            contexts.append(dict(start=chunk['start_frame'],end=chunk['end_frame_exclusive'],**chunk['execution_context']))
            update(status='native_C_core_only_decoding',emitted=chunk['end_frame_exclusive'],total=len(rgb))
            yield chunk
    packed,maxima=collect_native_chunks(chunks(),tuple(rgb.shape[:3]),times)
    with (folder/'native_C.npz').open('xb') as f:
        np.savez_compressed(f,packed_masks=packed,frame_probabilities=maxima,source_time_s=times)
    base.publish(folder/'native_C_execution.json',dict(execution_protocol_sha256=sha(folder.parent/'execution_protocol.json'),
        checkpoint_sha256=p['model_bindings'][p['models']['native_C']],original_frames=len(rgb),
        original_shape=list(rgb.shape),contexts=contexts,threshold=.5,FP32=True))
    del m;torch.cuda.empty_cache()
    return packed,maxima


def observe(out,p,source,update,base):
    import numpy as np
    import run_flicker_multimodel_observed_inference as components
    folder=out/'observations';execution_sha=sha(out/'execution_protocol.json')
    if folder.exists():raise ValueError('Fresh observations required; old incomplete attempts preserved')
    folder.mkdir();base.publish(folder/'source_started.json',dict(protocol_sha256=sha(out/'protocol.json'),
        execution_protocol_sha256=execution_sha,source_sha256=p['source_sha256']))
    update(status='decode_all_original_frames')
    obs=components.create_observed_store(source['video_path'],folder/'source',source['video_sha256'],None)
    rgb=np.load(folder/'source/rgb.npy',mmap_mode='r',allow_pickle=False)
    base.validate_observation(obs,source,p['original_clock'],rgb)
    config={k:p[k] for k in (*base.CONFIG,'models','raft_producer')}
    upstream=components.upstream(folder,config,rgb,update)
    packed,maxima=native_mask(folder,p,rgb,obs,np.asarray(upstream['frame_probabilities'],np.float64),update,base)
    pooled=components.motion(folder,config,rgb,update)
    features=components.features(folder,config,rgb,np.asarray(upstream['frame_probabilities'],np.float64),packed,pooled,update)
    from prepare_flicker_new_train_gpu_observations import validate_outputs
    validate_outputs(folder,dict(clock=p['original_clock']),upstream,packed,maxima,pooled,features)
    base.verify({source['video_path']:source['video_sha256']})
    base.publish(folder/'completed.json',dict(completed=True,protocol_sha256=sha(out/'protocol.json'),
        execution_protocol_sha256=execution_sha,source_sha256=p['source_sha256'],video_sha256=source['video_sha256'],
        original_frames=obs['frame_count'],labels_used=False,paired_clean_used=False,
        native_execution_receipt_sha256=sha(folder/'native_C_execution.json'),
        files_sha256={n:sha(folder/n) for n in base.OBS_FILES}))
    base.observation_receipt(out,p,source)
    if read(folder/'features.json')['feature_names']!=p['feature_names']:raise ValueError('Feature schema changed')
    return obs,rgb,packed,np.load(folder/'features.npy',mmap_mode='r',allow_pickle=False)


def run(out):
    sys.path.insert(0,str(out/'code'))
    import numpy as np
    import torch
    import infer_flicker_retained_parent_cuda as base
    p,source,e=validate(out,base)
    def update(**values):
        if values.get('status')!='failed' and shutil.disk_usage(out).free<40*1024**3:
            raise RuntimeError('Keep40GiB disk reserve')
        value=dict(pid=os.getpid(),updated_at=base.utc(),**values)
        temporary=out/('status.partial_'+uuid.uuid4().hex);base.publish(temporary,value)
        os.replace(temporary,out/'status.json');print(json.dumps(value),flush=True)
    with (out/'execution_worker.lock').open('a+') as worker, LOCK.open('a+') as shared:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        update(status='waiting_shared_GPU_lock');fcntl.flock(shared,fcntl.LOCK_EX)
        started=time.monotonic()
        def expired(signum,frame):raise TimeoutError('Explicit1800-second per-video budget; no fallback')
        signal.signal(signal.SIGALRM,expired);signal.alarm(e['run_budget_seconds'])
        try:
            base.begin_run(out,False);base.disk_budget(p['original_clock'],out)
            base.configure_observation_cuda(p)
            obs,rgb,packed,features=observe(out,p,source,update,base)
            actual=base.configure_temporal_cuda()
            models={k:v.to('cuda:0') for k,v in base.load_heads_cpu(p).items()}
            with np.load(p['normalization_path'],allow_pickle=False) as z:mean,std=z['mean'].copy(),z['std'].copy()
            update(status='predicting_frozen_retained600_heads',original_frames=obs['frame_count'])
            prediction=base.predict_retained_parent(models['full'],models['frame_context'],features,mean,std,
                device='cuda:0',source_pts=obs['source_pts'],source_time_s=obs['source_time_s'],time_base_fraction=obs['time_base_fraction'])
            result,masks=base.selected_output(obs,p,prediction,packed)
            work=Path(tempfile.mkdtemp(prefix='result.partial_',dir=out))
            np.savez_compressed(work/'predicted_masks.npz',packed_masks=masks,native_shape=np.asarray(rgb.shape[:3],np.int64),
                source_pts=np.asarray(obs['source_pts'],np.int64),source_time_s=np.asarray(obs['source_time_s'],np.float64))
            arrays={kind+'_'+key:value for kind in ('full','frame_context') for key,value in prediction[kind].items()}
            np.savez_compressed(work/'temporal_predictions.npz',**arrays,frame_probabilities=prediction['frame_probabilities'],
                frame_positive=prediction['frame_positive'],source_pts=np.asarray(obs['source_pts'],np.int64),source_time_s=prediction['source_time_s'])
            result.update(protocol_sha256=sha(out/'protocol.json'),execution_protocol_sha256=sha(out/'execution_protocol.json'),
                execution_version=VERSION,heads=p['heads'],CUDA_configuration=actual,
                spatial_execution='Frozen oldC parameters; identical8core4halo encoding; core-frame native2D decode microbatch1',
                mask_artifact=dict(path='predicted_masks.npz',sha256=sha(work/'predicted_masks.npz'),bitorder='little'),
                temporal_artifact=dict(path='temporal_predictions.npz',sha256=sha(work/'temporal_predictions.npz')))
            base.publish(work/'prediction.json',result);update(status='rendering_original_speed_review')
            binding=dict(protocol_sha256=sha(out/'protocol.json'),execution_protocol_sha256=sha(out/'execution_protocol.json'),
                prediction_sha256=sha(work/'prediction.json'))
            rendered=base.review_cpu(work,obs,rgb,packed,prediction['frame_probabilities'],binding)
            base.finalize_review(work,rendered,p,binding)
            base.verify(p['model_bindings']);base.verify({source['video_path']:source['video_sha256']})
            base.publish(work/'receipt.json',dict(protocol_sha256=sha(out/'protocol.json'),source_sha256=p['source_sha256'],
                execution_protocol_sha256=sha(out/'execution_protocol.json'),selection=p['selection'],
                observations_complete_sha256=sha(out/'observations/completed.json'),original_frames=obs['frame_count'],
                artifacts={n:sha(work/n) for n in base.RESULT_FILES}))
            if (out/'result').exists():raise ValueError('Unexpected result appeared')
            work.rename(out/'result');base.commit_result(out,p,source)
            done=dict(completed=True,execution_protocol_sha256=sha(out/'execution_protocol.json'),
                result_pointer_sha256=sha(out/'result.json'),original_frames=obs['frame_count'],
                original_shape=list(rgb.shape),source_sha256=source['video_sha256'],labels_used=False,
                elapsed_seconds=time.monotonic()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved=torch.cuda.max_memory_reserved(),threshold=.5,cuda_memory_fraction=.65,
                base_code_and_weights_unchanged=True,all_original_frames=True)
            base.publish(out/'execution_completed.json',done);update(status='completed',**done)
        except BaseException as error:
            update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-started)
            raise
        finally:signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','run'),required=True)
    parser.add_argument('--video',type=Path);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.action=='prepare':
        if a.video is None:parser.error('Prepare requires one original video')
        prepare(a.video,a.output)
    else:
        if a.video is not None:parser.error('Run consumes only frozen source contract')
        run(a.output.resolve())
