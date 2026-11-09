
from fractions import Fraction

import numpy as np
import torch
from torch import nn

from flicker_component_ablation import AblatedStateHead, canonical_names, removed_indices
from flicker_video_context import VideoContextRoutingHead


CORE_FRAMES = 32
HALO_FRAMES = 6
THRESHOLD = .5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def retained_head_kind(model):
    
    if type(model) is AblatedStateHead:
        require(tuple(model.removed_channels) == (), 'Full head must retain all 51 channels')
        kind = 'full'
    elif type(model) is VideoContextRoutingHead:
        require(model.context_mode == 'frame_context' and model.mode == 'learned_mix',
                'Only frame_context learned routing is locally chunkable')
        require(tuple(model.removed_channels) == removed_indices(canonical_names(), 'no_raft'),
                'Unexpected routing evidence mask')
        kind = 'frame_context'
    else:
        raise ValueError('Only exact frozen full and frame_context classes are supported')
    require(all(not m.training for m in model.modules()), 'All model modules must be eval-only')
    require(all(not p.requires_grad and p.device.type == 'cpu' and p.dtype == torch.float32
                and torch.isfinite(p).all().item() for p in model.parameters()),
            'Frozen finite CPU FP32 parameters required')

    def conv(layer, cls, cin, cout, kernel, padding, dilation, groups=1):
        require(type(layer) is cls and layer.in_channels == cin and layer.out_channels == cout
                and layer.kernel_size == kernel and layer.padding == padding
                and layer.dilation == dilation and layer.stride == (1,)*len(kernel)
                and layer.groups == groups and layer.padding_mode == 'zeros',
                'Changed convolution geometry invalidates the frozen halo contract')

    conv(model.embed, nn.Conv3d, 51, 24, (1, 1, 1), (0, 0, 0), (1, 1, 1))
    conv(model.spatial, nn.Conv3d, 24, 24, (1, 3, 3), (0, 1, 1), (1, 1, 1), 24)
    conv(model.temporal1, nn.Conv3d, 24, 24, (3, 1, 1), (1, 0, 0), (1, 1, 1))
    conv(model.temporal4, nn.Conv3d, 24, 24, (3, 1, 1), (4, 0, 0), (4, 1, 1))
    conv(model.region, nn.Conv3d, 24, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1))
    for head, output in ((model.state, 1), (model.boundary, 2)):
        require(type(head) is nn.Sequential and len(head) == 3 and type(head[1]) is nn.GELU
                and head[1].approximate == 'none', 'Changed readout architecture')
        conv(head[0], nn.Conv1d, 48, 32, (3,), (1,), (1,))
        conv(head[2], nn.Conv1d, 32, output, (1,), (0,), (1,))
    if kind == 'frame_context':
        require(set(model.gates) == {'temporal1', 'temporal4', 'state', 'boundary'},
                'Changed routing gates')
        for name in ('temporal1', 'temporal4'):
            conv(model.gates[name], nn.Conv3d, 48, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1))
        for name in ('state', 'boundary'):
            conv(model.gates[name], nn.Conv1d, 64, 1, (1,), (0,), (1,))
    return kind


def iter_retained_head_logits(model, features, mean, std, *, core_frames=CORE_FRAMES):
    
    retained_head_kind(model)
    require(type(core_frames) is int and core_frames == CORE_FRAMES, 'Frozen core_frames must be 32')
    require(isinstance(features, np.ndarray) and features.ndim == 4 and features.shape[1] == 51
            and min(features.shape) > 0 and features.dtype in (np.float16, np.float32),
            'Complete floating T,51,h,w raw observation features required')
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    require(mean.shape == (51,) and std.shape == (51,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and (std > 0).all(), 'Frozen finite 51-channel normalization required')
    mu = torch.from_numpy(mean.copy())[None, :, None, None]
    sd = torch.from_numpy(std.copy())[None, :, None, None]
    n = len(features)
    for start in range(0, n, CORE_FRAMES):
        stop = min(n, start+CORE_FRAMES)
        left, right = max(0, start-HALO_FRAMES), min(n, stop+HALO_FRAMES)
        data = np.array(features[left:right], copy=True)
        require(np.isfinite(data).all(), 'Nonfinite source feature')
        with torch.inference_mode():
            prediction = model((torch.from_numpy(data).float()-mu)/sd)
            a, b = start-left, stop-left
            state = prediction['frame_logits'][a:b]
            boundary = prediction['boundary_logits'][a:b]
            require(state.shape == (stop-start,) and boundary.shape == (stop-start, 2)
                    and state.dtype == boundary.dtype == torch.float32
                    and torch.isfinite(state).all().item() and torch.isfinite(boundary).all().item(),
                    'Incomplete or nonfinite original-frame model output')
            result = dict(frame_logits=state.contiguous().numpy().copy(),
                          boundary_logits=boundary.contiguous().numpy().copy(),
                          frame_probabilities=state.sigmoid().numpy().copy(),
                          boundary_probabilities=boundary.sigmoid().numpy().copy())
        yield dict(start_frame=start, end_frame_exclusive=stop, **result)


def complete_retained_head(model, features, mean, std, *, core_frames=CORE_FRAMES):
    n = len(features)
    result = {key:np.empty((n, 2) if key.startswith('boundary') else (n,), np.float32)
              for key in ('frame_logits', 'boundary_logits', 'frame_probabilities', 'boundary_probabilities')}
    seen = 0
    for chunk in iter_retained_head_logits(model, features, mean, std, core_frames=core_frames):
        a, b = chunk['start_frame'], chunk['end_frame_exclusive']
        require(a == seen and a < b <= n, 'Missing, repeated, or out-of-order original output frame')
        for key in result:
            result[key][a:b] = chunk[key]
        seen = b
    require(seen == n, 'Incomplete source output coverage')
    return result


def native_clock(source_pts, source_time_s, time_base_fraction, count):
    
    pts = list(source_pts)
    require(len(pts) == count and count > 0
            and all(type(p) is int or isinstance(p, np.integer) and not isinstance(p, np.bool_) for p in pts),
            'Complete integer original PTS required')
    pts = [int(p) for p in pts]
    times = np.array(source_time_s, dtype=np.float64, copy=True)
    base = Fraction(time_base_fraction)
    require(base > 0 and times.shape == (count,) and np.isfinite(times).all()
            and all(b > a for a, b in zip(pts, pts[1:]))
            and np.allclose(times, [float(p*base) for p in pts], rtol=0, atol=1e-9),
            'Original PTS/time_base/source_time_s disagree')
    return dict(source_pts=pts, source_time_s=times, time_base_fraction=str(base))


def combine_retained_scores(full_probabilities, frame_context_probabilities):
    
    full = np.asarray(full_probabilities)
    frame = np.asarray(frame_context_probabilities)
    require(full.ndim == frame.ndim == 1 and full.shape == frame.shape and len(full) > 0
            and np.isfinite(full).all() and np.isfinite(frame).all()
            and ((full >= 0) & (full <= 1)).all() and ((frame >= 0) & (frame <= 1)).all(),
            'Complete paired finite frame probabilities required')
    accepted = bool(full.max() >= THRESHOLD)
    final = frame.copy() if accepted else np.zeros_like(frame)
    return dict(video_accepted=accepted, full_video_max_frame_probability=float(full.max()),
                frame_probabilities=final, frame_positive=final >= THRESHOLD, threshold=THRESHOLD)


def predict_retained_parent(full_model, frame_model, features, mean, std, *, source_pts,
                            source_time_s, time_base_fraction, core_frames=CORE_FRAMES):
    
    require(retained_head_kind(full_model) == 'full' and retained_head_kind(frame_model) == 'frame_context',
            'Supply full gate then frame_context, never swap heads')
    clock = native_clock(source_pts, source_time_s, time_base_fraction, len(features))
    full = complete_retained_head(full_model, features, mean, std, core_frames=core_frames)
    frame = complete_retained_head(frame_model, features, mean, std, core_frames=core_frames)
    combined = combine_retained_scores(full['frame_probabilities'], frame['frame_probabilities'])
    return dict(full=full, frame_context=frame, **combined, **clock,
                original_frames=len(features), temporal_strategy='full_gate_frame_context',
                boundary_used_for_decoding=False, time_sampling_stride=1,
                source_frames_dropped=False, source_fps_changed=False,
                empty_output_spatial_mask_veto=False,
                frozen_C_mask_fraction_is_input_feature=True,
                core_frames=CORE_FRAMES, halo_frames=HALO_FRAMES)
