
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PATCH = 14
LAGS = (1, 4, 8)
SEMANTIC_CHANNELS = 16


def pad_native(x, patch=PATCH):
    h, w = x.shape[-2:]
    return F.pad(x, (0, (-w) % patch, 0, (-h) % patch), mode='replicate')


def pool_native(x, patch=PATCH, kind='mean'):
    
    x = pad_native(x, patch)
    return F.avg_pool2d(x, patch, patch) if kind == 'mean' else F.max_pool2d(x, patch, patch)


def shifted(x, delta):
    
    n = len(x)
    index = torch.arange(n, device=x.device) + delta
    valid = (index >= 0) & (index < n)
    return x[index.clamp(0, n - 1)], valid


def semantic_evidence(embedding):
    
    if embedding.ndim != 4 or not torch.isfinite(embedding).all():
        raise ValueError('Complete finite per-frame embeddings required')
    z = F.normalize(embedding.float(), dim=1)
    generator = torch.Generator(device='cpu').manual_seed(20261007)
    projection, _ = torch.linalg.qr(torch.randn(z.shape[1], SEMANTIC_CHANNELS, generator=generator))
    projected = torch.einsum('tchw,ck->tkhw', z, projection.to(z.device))
    
    features = [projected * math.sqrt(z.shape[1])]
    names = [f'dino_projection_{i}' for i in range(SEMANTIC_CHANNELS)]
    for lag in LAGS:
        for direction in (-1, 1):
            ref, valid = shifted(z, direction * lag)
            flag = valid[:, None, None, None].expand(-1, 1, z.shape[2], z.shape[3])
            distance = (1 - (z * ref).sum(1, keepdim=True)).clamp(0, 2) * flag
            features.extend((distance, flag.float()))
            names.extend((f'dino_cosine_{direction * lag}', f'dino_available_{direction * lag}'))
    return torch.cat(features, dim=1), names


@torch.inference_mode()
def dino_embeddings(model, rgb, device='cuda', progress=None):
    
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError('Original RGB T,H,W,3 uint8 required')
    mean = torch.tensor((.485, .456, .406), device=device)[None, :, None, None]
    std = torch.tensor((.229, .224, .225), device=device)[None, :, None, None]
    result = []
    for t in range(len(rgb)):
        x = torch.from_numpy(rgb[t].copy()).permute(2, 0, 1)[None].to(device).float() / 255
        x = (pad_native(x) - mean) / std
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
            tokens = model(pixel_values=x).last_hidden_state[:, 1:]
        gh, gw = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        if tokens.shape[1] != gh * gw:
            raise ValueError('DINO patch count differs from original padded source')
        result.append(tokens[0].transpose(0, 1).reshape(-1, gh, gw).float().cpu().half())
        if progress is not None and (t % 10 == 0 or t + 1 == len(rgb)):
            progress(t + 1, len(rgb))
    return torch.stack(result)


@torch.inference_mode()
def make_observation_features(rgb, embedding, r3d_scores, native_baseline_masks, motion_reader, progress=None):
    
    n, h, w, _ = rgb.shape
    if len(embedding) != n or len(r3d_scores) != n or native_baseline_masks.shape != (n, h, w):
        raise ValueError('Incomplete evidence source')
    semantic, names = semantic_evidence(embedding)
    gh, gw = semantic.shape[-2:]
    images, stds, rawdiff, motion, baseline = [], [], [], [], []
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
        cues = torch.from_numpy(motion_reader.chunk(t, t + 1))
        
        pooled_motion = torch.cat((pool_native(cues), pool_native(cues[:, :1], kind='max')), dim=1)
        bm = torch.from_numpy(native_baseline_masks[t].copy())[None, None].float()
        images.append(mu[0]); stds.append(sigma[0]); rawdiff.append(torch.cat(changes, 1)[0])
        motion.append(pooled_motion[0]); baseline.append(pool_native(bm)[0])
        if progress is not None and (t % 20 == 0 or t + 1 == n):
            progress(t + 1, n)
    appearance = torch.stack(images)
    parts = [semantic, appearance, torch.stack(stds), torch.stack(rawdiff), torch.stack(motion), torch.stack(baseline)]
    names += ['rgb_mean_r', 'rgb_mean_g', 'rgb_mean_b', 'rgb_std_r', 'rgb_std_g', 'rgb_std_b',
              'raw_previous_mean', 'raw_previous_max', 'raw_next_mean', 'raw_next_max',
              'raft_aligned_mean', 'raft_support', 'raft_consistent', 'raft_available', 'raft_aligned_max',
              'frozen_C_mask_fraction']
    for lag in LAGS:
        for direction in (-1, 1):
            ref, flag = shifted(appearance, direction * lag)
            parts.append((appearance - ref).abs().mean(1, keepdim=True) * flag[:, None, None, None])
            names.append(f'rgb_patch_difference_{direction * lag}')
    scores = torch.as_tensor(r3d_scores, dtype=torch.float32).reshape(n, 1, 1, 1).expand(n, 1, gh, gw)
    parts.append(scores); names.append('frozen_r3d_context')
    features = torch.cat(parts, dim=1)
    if features.shape[1] != len(names) or not torch.isfinite(features).all():
        raise ValueError('Invalid multimodel features')
    return features.half().numpy(), names


class DenseStateHead(nn.Module):
    
    def __init__(self, channels, width=24):
        super().__init__()
        self.embed = nn.Conv3d(channels, width, 1)
        self.spatial = nn.Conv3d(width, width, (1, 3, 3), padding=(0, 1, 1), groups=width)
        self.temporal1 = nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0))
        self.temporal4 = nn.Conv3d(width, width, (3, 1, 1), padding=(4, 0, 0), dilation=(4, 1, 1))
        self.region = nn.Conv3d(width, 1, 1)
        self.state = nn.Sequential(nn.Conv1d(width * 2, 32, 3, padding=1), nn.GELU(), nn.Conv1d(32, 1, 1))
        self.boundary = nn.Sequential(nn.Conv1d(width * 2, 32, 3, padding=1), nn.GELU(), nn.Conv1d(32, 2, 1))

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError('One full video T,C,h,w required')
        x = features.permute(1, 0, 2, 3)[None]
        x = F.gelu(self.embed(x))
        x = x + F.gelu(self.spatial(x))
        x = x + F.gelu(self.temporal1(x))
        x = x + F.gelu(self.temporal4(x))
        flat = x.flatten(3)
        local = flat.topk(min(8, flat.shape[-1]), dim=-1).values.mean(-1)
        pooled = torch.cat((flat.mean(-1), local), dim=1)
        return {'frame_logits': self.state(pooled)[0, 0],
                'boundary_logits': self.boundary(pooled)[0].transpose(0, 1),
                'region_logits': self.region(x)[0, 0]}


def balanced_bce(logits, target):
    
    target = target.to(logits.dtype)
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    positive = target > 0
    if positive.any() and (~positive).any():
        return .5 * (raw[positive].mean() + raw[~positive].mean())
    return raw.mean()


def boundary_targets(y):
    y = y.bool()
    previous = F.pad(y[:-1], (1, 0), value=False)
    following = F.pad(y[1:], (0, 1), value=False)
    return torch.stack((y & ~previous, y & ~following), dim=-1).float()


def supervision_loss(prediction, frame_target, grid_target):
    
    state = balanced_bce(prediction['frame_logits'], frame_target)
    boundary = balanced_bce(prediction['boundary_logits'], boundary_targets(frame_target))
    region = balanced_bce(prediction['region_logits'], grid_target)
    return state + .2 * boundary + .2 * region, {'state': float(state.detach()),
        'boundary': float(boundary.detach()), 'region': float(region.detach())}
