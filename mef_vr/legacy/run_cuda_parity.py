
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import fcntl, hashlib, json, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
BASE=Path('/mnt/e/VRFlickerLab/external_benchmarks/preparation_20261014_v1/line2_ours_xr_native_pilot_v2/VR_reference_Level001')
LOCK=Path('/mnt/e/VRFlickerLab/external_benchmarks/preparation_20261014_v1/gpu_evaluation.lock')
VIDEO=Path('/mnt/d/VRFlickerLab/scene_library/production/object_motion_3000_run_001/scenes/ai2thor_floorplan_005/videos/ai2thor_floorplan_005__reserve_04__cfg01_low_tint_subtle/training.mp4')
sys.path.insert(0,str(BASE/'code'))
from infer_flicker_retained_parent_cuda import read,sha,verify,publish,utc,configure_observation_cuda,configure_temporal_cuda
from r3d_native_motion_cues import NativeMotionCueRefiner
from r3d_native_zero_motion_inference import predict_zero_motion_chunks
from r3d_native_pixel_refiner import temporal_chunks
from native_core_decoder import predict_native_core_chunks

def main():
    out=ROOT/'cuda720p_parity_v1';out.mkdir(exist_ok=False)
    p=read(BASE/'protocol.json')
    verify({str(BASE/'code'/n):s for n,s in p['code_sha256'].items()})
    weight=p['models']['native_C'];verify({weight:p['model_bindings'][weight]})
    verify({str(VIDEO):'e205ea55b6bc46ce452e3524c3697266587c48f5c70ed4c5d2872fc7237b8177'})
    cap=cv2.VideoCapture(str(VIDEO));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release();rgb=np.stack(frames);del frames
    if rgb.shape != (120,720,1280,3) or fps != 30:raise ValueError('Train video contract changed')
    
    upstream=np.linspace(.1,.9,len(rgb),dtype=np.float64)
    times=np.arange(len(rgb),dtype=np.float64)/fps
    contract=dict(created_at=utc(),video=str(VIDEO),video_sha256=sha(VIDEO),shape=list(rgb.shape),fps=fps,
        checkpoint=weight,checkpoint_sha256=sha(weight),old_code_sha256=p['code_sha256'],
        adapter_sha256=sha(ROOT/'native_core_decoder.py'),test_sha256=sha(__file__),
        original_frames=120,source_time_s=times.tolist(),upstream='fixed deterministic linear .1-.9 engineering fixture; no task metrics',
        cuda_memory_fraction=.65,rtol=2e-5,atol=2e-6,maximum_allowed_binary_flips=0,
        no_resize=True,no_frame_sampling=True,FP32=True,core=8,halo=4,
        native_decode_microbatch=1,budget_seconds=900)
    publish(out/'contract.json',contract)
    with LOCK.open('a+') as lock:
        print('WAIT_GPU_LOCK',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
        started=time.monotonic();configure_observation_cuda(p);configure_temporal_cuda()
        model=NativeMotionCueRefiner(False).cuda().eval().requires_grad_(False)
        checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
        if checkpoint['arm']!='C_zero_motion' or checkpoint['step']!=300:raise ValueError('Wrong checkpoint')
        model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
        old=predict_zero_motion_chunks(model,rgb,upstream,fps,times,device='cuda',return_logits=True)
        new=predict_native_core_chunks(model,rgb,upstream,fps,times,device='cuda',return_logits=True)
        rows=[];contexts=list(temporal_chunks(120,8,4))
        for index,(a,b) in enumerate(zip(old,new,strict=True)):
            torch.cuda.synchronize()
            if time.monotonic()-started>900:raise TimeoutError('15 minute parity budget exceeded')
            x,y=a['pixel_logits'],b['pixel_logits'];diff=np.abs(x-y)
            left,right,start,end=contexts[index]
            if (a['start_frame'],a['end_frame_exclusive'])!=(left+start,left+end):raise ValueError('Old context')
            if (b['start_frame'],b['end_frame_exclusive'])!=(left+start,left+end):raise ValueError('New context')
            if b['execution_context']!=dict(left=left,right=right,core_start=start,core_end=end):raise ValueError('Halo changed')
            if not np.array_equal(a['source_time_s'],b['source_time_s']):raise ValueError('Clock changed')
            flips=np.count_nonzero((a['pixel_probabilities']>=.5)!=(b['pixel_probabilities']>=.5))
            row=dict(chunk=index,start=left+start,end=left+end,context=[left,right,start,end],
                max_abs_error=float(diff.max()),mean_abs_error=float(diff.mean()),
                logit_bitwise_equal=bool(np.array_equal(x,y)),probability_bitwise_equal=bool(np.array_equal(a['pixel_probabilities'],b['pixel_probabilities'])),
                close=bool(np.allclose(x,y,rtol=2e-5,atol=2e-6)),binary_flips=int(flips),pixel_count=x.size)
            rows.append(row);publish(out/f'chunk_{index:02d}.json',row)
            print(json.dumps(row),flush=True)
        report=dict(completed=True,passed=all(r['close'] and r['binary_flips']==0 for r in rows),
            total_frames=sum(r['end']-r['start'] for r in rows),total_pixels=sum(r['pixel_count'] for r in rows),
            binary_flips=sum(r['binary_flips'] for r in rows),max_abs_error=max(r['max_abs_error'] for r in rows),
            bitwise_equal=all(r['logit_bitwise_equal'] for r in rows),complete_original_clock_and_context=True,
            elapsed_seconds=time.monotonic()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved=torch.cuda.max_memory_reserved(),contract_sha256=sha(out/'contract.json'),
            torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            classifier_efficacy_test=False,rows=rows)
        publish(out/'report.json',report);print(json.dumps(report),flush=True)
        if not report['passed']:raise RuntimeError('Parity failed; no production promotion')

if __name__=='__main__':main()
