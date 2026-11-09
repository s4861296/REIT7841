
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
FROZEN = Path('/mnt/e/VRFlickerLab/model_evaluation/real_gt_three_models_20261014_v1/ours/frozen_code')
sys.path.insert(0,str(FROZEN))
import numpy as np
import torch
import native_core_decoder as decoder
from r3d_native_pixel_data import feature_chunk as original
from r3d_native_pixel_refiner import temporal_chunks
from r3d_native_motion_cues import NativeMotionCueRefiner
import bounded_features as adapter

LOCK = Path('/tmp/vrflicker_real_gt_three_models_20261014_gpu.lock')
original_core = decoder.core_logits_cpu
ADAPTER_SHA256 = adapter.sha(adapter.__file__)
MEASUREMENT_SHA256 = adapter.sha(__file__)


def source(n,h,w):
    rng=np.random.default_rng(90517+n+h+w)
    return dict(rgb=rng.integers(0,256,(n,h,w,3),dtype=np.uint8),
        logits=np.linspace(-13.815509,13.815509,n,dtype=np.float32))


def sourcebinding(data):
    n,h,w,_=data['rgb'].shape
    return dict(n=n,h=h,w=w,seed=90517+n+h+w,
        rgb_sha256=hashlib.sha256(data['rgb'].tobytes()).hexdigest(),
        upstream_logits_sha256=hashlib.sha256(data['logits'].tobytes()).hexdigest())


def checkpoint_model(device):
    p=json.loads((FROZEN.parent/'protocol.json').read_text())
    path=Path(p['models']['native_C'])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==p['model_bindings'][str(path)]
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    assert checkpoint['arm']=='C_zero_motion' and checkpoint['step']==300
    model=NativeMotionCueRefiner(False).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'],strict=True)
    return model,digest


def binary_flips(left,right,device):
    
    
    left_binary=torch.from_numpy(left).to(device).sigmoid()>=.5
    right_binary=torch.from_numpy(right).to(device).sigmoid()>=.5
    return int(torch.count_nonzero(left_binary!=right_binary).item())


def run(device):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    if device=='cuda':
        torch.cuda.set_per_process_memory_fraction(.65)
    receipt=adapter._install_for_parity(FROZEN)
    assert decoder.feature_chunk is adapter.bounded_feature_chunk
    assert decoder.core_logits_cpu is adapter.bounded_core_logits_cpu
    assert original is not decoder.feature_chunk
    cases=[]
    fixtures=[]
    with torch.inference_mode():
        for n,h,w in ((1,9,13),(2,8,7),(21,13,17),(33,24,31)):
            data=source(n,h,w)
            fixtures.append(sourcebinding(data))
            for left,right,start,end in temporal_chunks(n,8,4):
                expected=original(data,left,right,59.94,device=device)
                actual=adapter.bounded_feature_chunk(data,left,right,59.94,device=device)
                assert expected.shape==actual.shape and torch.equal(expected,actual)
                cases.append(dict(n=n,shape=[h,w],context=[left,right,start,end],bitwise_equal=True))
                del expected,actual
        for invalid in ((-1,1),(0,0),(0,22)):
            try: adapter.bounded_feature_chunk(source(21,13,17),*invalid,60.,device=device)
            except ValueError: pass
            else: raise AssertionError('Invalid interval accepted')
        model,checkpoint_sha=checkpoint_model(device)
        model_cases=[]
        for n,h,w,left,right,start,end in ((1,13,17,0,1,0,1),
                (21,37,53,4,20,4,12),(33,64,80,20,33,4,12)):
            data=source(n,h,w)
            fixtures.append(sourcebinding(data))
            old=original(data,left,right,60.,device=device)
            new=adapter.bounded_feature_chunk(data,left,right,60.,device=device)
            cues=old.new_zeros((1,right-left,4,h,w))
            left_logits=original_core(model,old[None],cues,start,end)
            right_logits=decoder.core_logits_cpu(model,new[None],cues,start,end)
            np.testing.assert_allclose(left_logits,right_logits,
                atol=adapter.PARITY_ATOL,rtol=adapter.PARITY_RTOL)
            flips=binary_flips(left_logits,right_logits,device)
            assert flips==0
            model_cases.append(dict(n=n,shape=[h,w],context=[left,right,start,end],
                bitwise_equal=bool(np.array_equal(left_logits,right_logits)),
                max_absolute_error=float(np.max(np.abs(left_logits-right_logits))),
                mean_absolute_error=float(np.mean(np.abs(left_logits-right_logits))),
                binary_flips=flips,logits_close=True))
            del old,new,cues
        del model
        memory=None
        if device=='cuda':
            data=source(21,384,512)
            fixtures.append(sourcebinding(data))
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            old=original(data,4,20,60.,device=device);torch.cuda.synchronize()
            old_peak=torch.cuda.max_memory_allocated();del old
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            new=adapter.bounded_feature_chunk(data,4,20,60.,device=device);torch.cuda.synchronize()
            new_peak=torch.cuda.max_memory_allocated();del new
            assert new_peak<old_peak
            feature_memory=dict(original_peak_bytes=old_peak,bounded_peak_bytes=new_peak)
            model,_=checkpoint_model(device)
            features=adapter.bounded_feature_chunk(data,4,20,60.,device=device)[None]
            cues=features.new_zeros((1,16,4,384,512))
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            old_logits=original_core(model,features,cues,4,12);torch.cuda.synchronize()
            old_peak=torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            new_logits=decoder.core_logits_cpu(model,features,cues,4,12);torch.cuda.synchronize()
            new_peak=torch.cuda.max_memory_allocated()
            np.testing.assert_allclose(old_logits,new_logits,
                atol=adapter.PARITY_ATOL,rtol=adapter.PARITY_RTOL)
            assert binary_flips(old_logits,new_logits,device)==0
            assert new_peak<old_peak
            memory=dict(feature_assembly=feature_memory,spatial_encoder=dict(
                original_peak_bytes=old_peak,bounded_peak_bytes=new_peak,
                shape=[16,384,512],max_absolute_error=float(np.max(np.abs(old_logits-new_logits))),
                binary_flips=0))
            del features,cues,model
    assert adapter.sha(adapter.__file__)==ADAPTER_SHA256
    assert adapter.sha(__file__)==MEASUREMENT_SHA256
    return dict(passed=True,device=device,version=adapter.VERSION,
        adapter_sha256=ADAPTER_SHA256,
        measurement_script_sha256=MEASUREMENT_SHA256,
        source_files=adapter.EXPECTED_SHA256,frozen_source_sha256=adapter.EXPECTED_SHA256,
        checkpoint_path=str(adapter.CHECKPOINT),checkpoint_sha256=checkpoint_sha,
        sourcebinding=dict(kind='deterministic_generated_test_input',seed_rule='90517+n+h+w',
            label_blind=True,gt_read=False,fixtures=fixtures),
        shared_gpu_lock=str(LOCK) if device=='cuda' else None,
        temporal_core_frames=8,temporal_halo_frames=4,
        feature_cases=cases,all_features_bitwise_equal=True,checkpoint_cases=model_cases,
        actual_checkpoint_logits_bitwise_equal=all(x['bitwise_equal'] for x in model_cases),
        actual_checkpoint_logits_close=True,binary_flips=0,binary_rule='same-device FP32 sigmoid >= 0.5',
        parity_atol=adapter.PARITY_ATOL,parity_rtol=adapter.PARITY_RTOL,
        memory=memory,installation=receipt,original_files_modified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    if args.device=='cuda':
        print(json.dumps({'status':'waiting_shared_gpu_lock','path':str(LOCK)}),flush=True)
    with (LOCK.open('a') if args.device=='cuda' else nullcontext(None)) as handle:
        if handle is not None: fcntl.flock(handle,fcntl.LOCK_EX)
        result=run(args.device)
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
