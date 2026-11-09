
from __future__ import annotations
import numpy as np
import torch
from flicker_multimodel_core import LAGS, pool_native, semantic_evidence, shifted


@torch.inference_mode()
def pooled_motion_cues(aligned, support, consistent, has_previous):
    if aligned.dtype != np.float32 or support.dtype != np.bool_ or consistent.dtype != np.bool_:
        raise ValueError('Native FP32 residual and boolean geometry required')
    if aligned.shape != support.shape or support.shape != consistent.shape or aligned.ndim != 2:
        raise ValueError('Same original image coordinates required')
    if not np.isfinite(aligned).all() or (aligned < 0).any() or (aligned > 1).any() or np.any(consistent & ~support):
        raise ValueError('Invalid geometric evidence')
    if not has_previous and (aligned.any() or support.any() or consistent.any()):
        raise ValueError('Frame0 must retain explicit missing predecessor')
    cues = np.stack((aligned * support, support, consistent, np.full(aligned.shape, bool(has_previous)))).astype(np.float32)
    x = torch.from_numpy(cues)[None]
    return torch.cat((pool_native(x), pool_native(x[:, :1], kind='max')), dim=1)[0].numpy()


@torch.inference_mode()
def assemble_observed_features(rgb, embedding, r3d_scores, predicted_native_masks, motion_pooled):
    n, h, w, _ = rgb.shape
    semantic, names = semantic_evidence(embedding)
    gh, gw = semantic.shape[-2:]
    if (predicted_native_masks.shape != (n, h, w) or motion_pooled.shape != (n, 5, gh, gw)
            or len(r3d_scores) != n or rgb.dtype != np.uint8): raise ValueError('Incomplete original evidence')
    images, stds, raw, masks = [], [], [], []
    for t in range(n):
        current = torch.from_numpy(rgb[t].copy()).permute(2, 0, 1)[None].float() / 255
        mu = pool_native(current)
        sigma = (pool_native(current.square()) - mu.square()).clamp_min(0).sqrt()
        changes = []
        for delta in (-1, 1):
            j = min(max(t + delta, 0), n - 1)
            other = torch.from_numpy(rgb[j].copy()).permute(2, 0, 1)[None].float() / 255
            diff = (current - other).abs().mean(1, keepdim=True)
            changes.extend((pool_native(diff), pool_native(diff, kind='max')))
        mask = torch.from_numpy(predicted_native_masks[t].copy())[None, None].float()
        images.append(mu[0]); stds.append(sigma[0]); raw.append(torch.cat(changes, 1)[0]); masks.append(pool_native(mask)[0])
    appearance = torch.stack(images)
    parts = [semantic, appearance, torch.stack(stds), torch.stack(raw), torch.from_numpy(np.array(motion_pooled, copy=True)), torch.stack(masks)]
    names += ['rgb_mean_r', 'rgb_mean_g', 'rgb_mean_b', 'rgb_std_r', 'rgb_std_g', 'rgb_std_b',
        'raw_previous_mean', 'raw_previous_max', 'raw_next_mean', 'raw_next_max',
        'raft_aligned_mean', 'raft_support', 'raft_consistent', 'raft_available', 'raft_aligned_max', 'frozen_C_mask_fraction']
    for lag in LAGS:
        for direction in (-1, 1):
            ref, flag = shifted(appearance, direction * lag)
            parts.append((appearance - ref).abs().mean(1, keepdim=True) * flag[:, None, None, None])
            names.append(f'rgb_patch_difference_{direction * lag}')
    parts.append(torch.as_tensor(r3d_scores, dtype=torch.float32).reshape(n, 1, 1, 1).expand(n, 1, gh, gw))
    names.append('frozen_r3d_context')
    features = torch.cat(parts, dim=1)
    if features.shape[1] != len(names) or not torch.isfinite(features).all(): raise ValueError('Nonfinite observed features')
    return features.half().numpy(), names
