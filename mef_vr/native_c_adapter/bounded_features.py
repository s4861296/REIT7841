
from pathlib import Path
import hashlib
import importlib
import json

import numpy as np
import torch
import torch.nn.functional as F

FROZEN_CODE = Path('/mnt/e/VRFlickerLab/model_evaluation/real_gt_three_models_20261014_v1/ours/frozen_code')
CHECKPOINT = Path('/mnt/d/VRFlickerLab/model_training/r3d18_flicker_v1/native_motion_input_F100_pairing_recovery_20260925_v3/trial/formal/C_zero_motion/checkpoint-0003/training_state.pt')
CHECKPOINT_SHA256 = '5cbbd8c4cbe5efa30a29115a3b27ed2b594e8607613fe9379f14374cc3229a98'
VERSION = 'native_c_bounded_features_framewise_encoder_v2'
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-5
EXPECTED_SHA256 = {
    'native_core_decoder.py': '2ee98b597a2e179ea447e7404d671c807b0fded0252629bf0273dc9472d491e2',
    'r3d_native_pixel_data.py': 'acc2cbcd5c92fe57486c337504ed5a901184d37dffe341fe5c2f96232dfd05b5',
    'r3d_native_pixel_refiner.py': 'e22ff99ccc2976f9d3c1ebf40d1175ad26ca0155c557abc974da65b11c5bef93',
    'r3d_native_motion_cues.py': '092a33df97828aa5abca29af15088076fd94e3770510a6d33200e3583b2f0704',
    'r3d_native_restoration_aux.py': '0cbe4e1180039e21ae27a7a50f91bef1d5dfef2ee98dea4d86f24f4da5400c7d',
}
_original = None
_original_decoder = None
_active_frozen = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _frozen_modules(frozen_code_dir):
    frozen = Path(frozen_code_dir).resolve()
    for name, digest in EXPECTED_SHA256.items():
        if sha(frozen/name) != digest:
            raise ValueError('Frozen source changed: '+name)
    modules = {}
    for filename in EXPECTED_SHA256:
        module = importlib.import_module(Path(filename).stem)
        if Path(module.__file__).resolve() != frozen/filename:
            raise ValueError('Refusing a non-frozen dependency: '+module.__file__)
        modules[Path(filename).stem] = module
    return modules['native_core_decoder'], modules['r3d_native_pixel_data']


def bounded_feature_chunk(data, left, right, fps, device='cuda'):
    
    if _original is None:
        raise RuntimeError('Install the explicitly bound adapter first')
    rgb, logits = data['rgb'], data['logits']
    n = len(rgb)
    if not 0 <= left < right <= n or logits.shape != (n,):
        raise ValueError('Invalid source chunk')
    h, w = rgb.shape[1:3]
    output = torch.empty((right-left,22,h,w),device=device,dtype=torch.float32)
    for index in range(left,right):
        frame_features = _original(data,index,index+1,fps,device=device)
        if frame_features.shape != (1,22,h,w) or frame_features.dtype != output.dtype:
            raise ValueError('Unexpected frozen feature geometry or precision')
        output[index-left:index-left+1].copy_(frame_features)
        del frame_features
    return output


def bounded_core_logits_cpu(model, features, cues, start, end):
    
    if model.use_native_skips is not False or features.dtype != torch.float32:
        raise ValueError('Only frozen C without native skips, FP32 supported')
    b,t,c,h,w = features.shape
    if b != 1 or c != 22 or cues.shape != (b,t,4,h,w):
        raise ValueError('Original single-video 22+4 channels required')
    if not (0 <= start < end <= t) or any(m.training for m in model.modules()):
        raise ValueError('Valid original core and eval model required')
    if cues.dtype != features.dtype or cues.device != features.device:
        raise ValueError('Same-device FP32 cues required')
    coarse = None
    for i in range(t):
        fine = model.stem(features[:,i])
        fine = fine + model.motion_projection(cues[:,i])
        mid = model.down2(fine)
        one_coarse = model.down4(mid)
        if coarse is None:
            mh,mw = mid.shape[-2:]
            ch,cw = one_coarse.shape[-2:]
            coarse = one_coarse.new_empty((b*t,32,ch,cw))
        coarse[i:i+1].copy_(one_coarse)
        del fine,mid,one_coarse
    
    sequence = coarse.reshape(b,t,32,ch,cw).permute(0,2,1,3,4)
    sequence = sequence + model.temporal(sequence)
    coarse = sequence.permute(0,2,1,3,4).reshape(b*t,32,ch,cw)
    del sequence
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


def _install_for_parity(frozen_code_dir):
    
    global _original, _original_decoder, _active_frozen
    decoder, data = _frozen_modules(frozen_code_dir)
    if decoder.feature_chunk not in (data.feature_chunk,bounded_feature_chunk):
        raise ValueError('Unexpected pre-existing native_C feature patch')
    if decoder.core_logits_cpu is bounded_core_logits_cpu:
        raise ValueError('Native C adapter already installed; refuse stacked patches')
    if decoder.core_logits_cpu.__module__ != 'native_core_decoder':
        raise ValueError('Unexpected pre-existing native_C decoder patch')
    _original = data.feature_chunk
    _original_decoder = decoder.core_logits_cpu
    _active_frozen = Path(frozen_code_dir).resolve()
    decoder.feature_chunk = bounded_feature_chunk
    decoder.core_logits_cpu = bounded_core_logits_cpu
    return dict(version=VERSION,patched=['native_core_decoder.feature_chunk','native_core_decoder.core_logits_cpu'],
        frozen_code_dir=str(_active_frozen),
        feature_implementation='unchanged frozen feature_chunk called once per original frame',
        allocation='preallocated complete original temporal chunk on requested device',
        spatial_encoder_microbatch_frames=1,temporal_sequence='complete original chunk unchanged',
        temporal_core_frames=8,temporal_halo_frames=4,channels=22,dtype='float32',
        precision_changed=False,weights_changed=False,threshold_changed=False,
        source_resize=False,frame_sampling=False,source_files=EXPECTED_SHA256)


def install(frozen_code_dir, *, parity_receipt):
    
    receipt = json.loads(Path(parity_receipt).read_text())
    if (receipt.get('passed') is not True or receipt.get('device') != 'cuda'
            or receipt.get('version') != VERSION
            or receipt.get('adapter_sha256') != sha(__file__)
            or receipt.get('measurement_script_sha256') != sha(Path(__file__).with_name('test_bounded_features.py'))
            or receipt.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or sha(CHECKPOINT) != CHECKPOINT_SHA256
            or receipt.get('source_files') != EXPECTED_SHA256
            or receipt.get('frozen_source_sha256') != EXPECTED_SHA256
            or receipt.get('all_features_bitwise_equal') is not True
            or receipt.get('actual_checkpoint_logits_close') is not True
            or receipt.get('binary_flips') != 0
            or receipt.get('parity_atol') != PARITY_ATOL
            or receipt.get('parity_rtol') != PARITY_RTOL
            or receipt.get('temporal_core_frames') != 8
            or receipt.get('temporal_halo_frames') != 4
            or receipt.get('shared_gpu_lock') != '/tmp/vrflicker_real_gt_three_models_20261014_gpu.lock'):
        raise ValueError('Measured GPU parity receipt missing or changed')
    result = _install_for_parity(frozen_code_dir)
    result.update(adapter_sha256=sha(__file__),checkpoint_sha256=CHECKPOINT_SHA256,
                  parity_receipt_sha256=sha(parity_receipt))
    return result
