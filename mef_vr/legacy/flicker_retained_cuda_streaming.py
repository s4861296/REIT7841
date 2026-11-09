
from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from flicker_component_ablation import AblatedStateHead, canonical_names, removed_indices
from flicker_video_context import VideoContextRoutingHead
from flicker_retained_parent_inference import combine_retained_scores, native_clock

CORE_FRAMES = 32
HALO_FRAMES = 6
MAX_DEVICE_FRAMES = CORE_FRAMES+2*HALO_FRAMES
CUDA_CONFIGURATION = dict(cublas_workspace_config=':4096:8', deterministic_algorithms=True,
    deterministic_warn_only=False, cuda_matmul_tf32=False, cudnn_tf32=False,
    cudnn_benchmark=False, cudnn_deterministic=True)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def explicit_device(device):
    
    require(isinstance(device, (str, torch.device)), 'Explicit cpu or cuda:index device required')
    selected = torch.device(device)
    require(selected.type in ('cpu', 'cuda')
        and (selected.type == 'cpu' and selected.index is None
             or selected.type == 'cuda' and selected.index is not None),
        'Explicit cpu or indexed cuda device required; no fallback')
    return selected


def check_cuda_configuration(config):
    
    require(set(config) == set(CUDA_CONFIGURATION)
        and all(type(config[k]) is type(expected) and config[k] == expected
                for k, expected in CUDA_CONFIGURATION.items()),
        'Restore exact deterministic FP32 CUDA head settings after GPU observation stages')
    return dict(config)


def current_cuda_configuration():
    
    
    return dict(cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic)


def retained_head_kind(model, *, device):
    
    selected = explicit_device(device)
    if type(model) is AblatedStateHead:
        require(tuple(model.removed_channels) == (), 'Full gate must retain all51 channels')
        kind = 'full'
    elif type(model) is VideoContextRoutingHead:
        require(model.context_mode == 'frame_context' and model.mode == 'learned_mix',
                'Only frame_context learned routing supports this fixed local halo')
        require(tuple(model.removed_channels) == removed_indices(canonical_names(), 'no_raft'),
                'Routing evidence mask changed')
        kind = 'frame_context'
    else:
        raise ValueError('Only exact full and frame_context classes are supported')
    require(all(not m.training for m in model.modules()), 'Eval-only model required')
    parameters = list(model.parameters())
    require(parameters and all(not p.requires_grad and p.dtype == torch.float32 and p.device == selected
                               for p in parameters),
            'Frozen FP32 parameters must already occupy the explicit device; never move/fallback silently')
    if selected.type == 'cuda': check_cuda_configuration(current_cuda_configuration())
    require(all(torch.isfinite(p).all().item() for p in parameters), 'Finite model parameters required')

    def conv(layer, cls, cin, cout, kernel, padding, dilation, groups=1):
        require(type(layer) is cls and layer.in_channels == cin and layer.out_channels == cout
            and layer.kernel_size == kernel and layer.padding == padding and layer.dilation == dilation
            and layer.stride == (1,)*len(kernel) and layer.groups == groups and layer.padding_mode == 'zeros',
            'Changed convolution geometry invalidates halo6')

    conv(model.embed, nn.Conv3d, 51, 24, (1,1,1), (0,0,0), (1,1,1))
    conv(model.spatial, nn.Conv3d, 24, 24, (1,3,3), (0,1,1), (1,1,1), 24)
    conv(model.temporal1, nn.Conv3d, 24, 24, (3,1,1), (1,0,0), (1,1,1))
    conv(model.temporal4, nn.Conv3d, 24, 24, (3,1,1), (4,0,0), (4,1,1))
    conv(model.region, nn.Conv3d, 24, 1, (1,1,1), (0,0,0), (1,1,1))
    for head, channels in ((model.state,1), (model.boundary,2)):
        require(type(head) is nn.Sequential and len(head) == 3 and type(head[1]) is nn.GELU
            and head[1].approximate == 'none', 'Changed state/boundary readout')
        conv(head[0], nn.Conv1d, 48, 32, (3,), (1,), (1,))
        conv(head[2], nn.Conv1d, 32, channels, (1,), (0,), (1,))
    if kind == 'frame_context':
        require(set(model.gates) == {'temporal1','temporal4','state','boundary'}, 'Changed routing gates')
        for name in ('temporal1','temporal4'):
            conv(model.gates[name], nn.Conv3d, 48, 1, (1,1,1), (0,0,0), (1,1,1))
        for name in ('state','boundary'):
            conv(model.gates[name], nn.Conv1d, 64, 1, (1,), (0,), (1,))
    return kind


def iter_retained_head_logits(model, features, mean, std, *, device, core_frames=CORE_FRAMES):
    
    selected = explicit_device(device); retained_head_kind(model, device=selected)
    require(type(core_frames) is int and core_frames == CORE_FRAMES, 'Frozen core32 required')
    require(isinstance(features, np.ndarray) and features.ndim == 4 and features.shape[1] == 51
        and min(features.shape) > 0 and features.dtype in (np.float16,np.float32),
        'Complete floating native-time T,51,h,w raw features required')
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    require(mean.shape == std.shape == (51,) and np.isfinite(mean).all() and np.isfinite(std).all()
        and (std > 0).all(), 'Frozen finite51-channel normalization required')
    mu = torch.from_numpy(mean.copy())[None,:,None,None]
    sd = torch.from_numpy(std.copy())[None,:,None,None]
    n = len(features)
    for start in range(0, n, CORE_FRAMES):
        stop = min(n, start+CORE_FRAMES)
        left, right = max(0,start-HALO_FRAMES), min(n,stop+HALO_FRAMES)
        require(right-left <= MAX_DEVICE_FRAMES, 'Chunk exceeds bounded temporal extent')
        data = np.array(features[left:right], copy=True)
        require(np.isfinite(data).all(), 'Nonfinite native source chunk')
        
        
        normalized = (torch.from_numpy(data).float()-mu)/sd
        require(normalized.device.type == 'cpu' and normalized.dtype == torch.float32
            and torch.isfinite(normalized).all().item(), 'Finite CPU FP32 normalized chunk required')
        with torch.inference_mode(), torch.autocast(device_type=selected.type, enabled=False):
            prediction = model(normalized.to(selected))
            a, b = start-left, stop-left
            state = prediction['frame_logits'][a:b]
            boundary = prediction['boundary_logits'][a:b]
            require(state.shape == (stop-start,) and boundary.shape == (stop-start,2)
                and state.dtype == boundary.dtype == torch.float32 and state.device == boundary.device == selected
                and torch.isfinite(state).all().item() and torch.isfinite(boundary).all().item(),
                'Incomplete/nonfinite native-frame FP32 output')
            result = dict(frame_logits=state.contiguous().cpu().numpy().copy(),
                boundary_logits=boundary.contiguous().cpu().numpy().copy(),
                frame_probabilities=state.sigmoid().cpu().numpy().copy(),
                boundary_probabilities=boundary.sigmoid().cpu().numpy().copy())
        
        del data, normalized, prediction, state, boundary
        yield dict(start_frame=start, end_frame_exclusive=stop, **result)


def complete_retained_head(model, features, mean, std, *, device, core_frames=CORE_FRAMES):
    
    n = len(features)
    result = {key: np.empty((n,2) if key.startswith('boundary') else (n,), np.float32)
              for key in ('frame_logits','boundary_logits','frame_probabilities','boundary_probabilities')}
    seen = 0
    for chunk in iter_retained_head_logits(model, features, mean, std, device=device, core_frames=core_frames):
        start, stop = chunk['start_frame'], chunk['end_frame_exclusive']
        require(start == seen and start < stop <= n, 'Missing/repeated/out-of-order original frame')
        for key in result: result[key][start:stop] = chunk[key]
        seen = stop
    require(seen == n, 'Incomplete native output coverage')
    return result


def predict_retained_parent(full_model, frame_model, features, mean, std, *, device,
                            source_pts, source_time_s, time_base_fraction, core_frames=CORE_FRAMES):
    
    selected = explicit_device(device)
    require(retained_head_kind(full_model, device=selected) == 'full'
        and retained_head_kind(frame_model, device=selected) == 'frame_context', 'Full gate then frame_context required')
    clock = native_clock(source_pts, source_time_s, time_base_fraction, len(features))
    full = complete_retained_head(full_model, features, mean, std, device=selected, core_frames=core_frames)
    frame = complete_retained_head(frame_model, features, mean, std, device=selected, core_frames=core_frames)
    combined = combine_retained_scores(full['frame_probabilities'], frame['frame_probabilities'])
    return dict(full=full, frame_context=frame, **combined, **clock,
        original_frames=len(features), temporal_strategy='full_gate_frame_context',
        boundary_used_for_decoding=False, empty_output_spatial_mask_veto=False,
        frozen_C_mask_fraction_is_input_feature=True, time_sampling_stride=1,
        source_frames_dropped=False, source_fps_changed=False, source_coordinates_changed=False,
        normalization_device='cpu', normalization_dtype='float32', device=str(selected),
        inference_dtype='float32', amp_used=False, CPU_fallback=False,
        core_frames=CORE_FRAMES, halo_frames=HALO_FRAMES, max_device_input_frames=MAX_DEVICE_FRAMES,
        full_source_feature_transfer=False, whole_pipeline_RAM_bounded=False,
        backend_numeric_equivalence_claimed=False, model_efficacy_claimed=False,
        CUDA_configuration=current_cuda_configuration() if selected.type == 'cuda' else None)
