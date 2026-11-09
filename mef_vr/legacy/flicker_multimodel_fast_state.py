
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from flicker_multimodel_core import DenseStateHead, balanced_bce, boundary_targets

ARMS = {
    'A_continue': {'fast': False, 'run_balance': False},
    'B_fast': {'fast': True, 'run_balance': False},
    'C_run_balance': {'fast': False, 'run_balance': True},
    'D_fast_run_balance': {'fast': True, 'run_balance': True},
}


def spatial_summary(x):
    flat = x.flatten(3)
    return torch.cat((flat.mean(-1), flat.topk(min(8, flat.shape[-1]), dim=-1).values.mean(-1)), dim=1)


class FastResidualStateHead(DenseStateHead):
    
    def __init__(self, channels, width=24):
        super().__init__(channels, width)
        self.fast = nn.Sequential(nn.Conv1d(width * 2, 32, 1), nn.GELU(), nn.Conv1d(32, 1, 1))
        nn.init.zeros_(self.fast[-1].weight)
        nn.init.zeros_(self.fast[-1].bias)

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError('One complete T,C,h,w video required')
        x = F.gelu(self.embed(features.permute(1, 0, 2, 3)[None]))
        x = x + F.gelu(self.spatial(x))
        quick = self.fast(spatial_summary(x))[0, 0]
        x = x + F.gelu(self.temporal1(x))
        x = x + F.gelu(self.temporal4(x))
        pooled = spatial_summary(x)
        return {'frame_logits': self.state(pooled)[0, 0] + quick,
                'boundary_logits': self.boundary(pooled)[0].transpose(0, 1),
                'region_logits': self.region(x)[0, 0], 'fast_residual_logits': quick}


def initialized_model(channels, parent_state, fast, seed=20261008):
    torch.manual_seed(seed)
    model = FastResidualStateHead(channels) if fast else DenseStateHead(channels)
    result = model.load_state_dict(parent_state, strict=not fast)
    if fast and (set(result.missing_keys) != {'fast.0.weight', 'fast.0.bias', 'fast.2.weight', 'fast.2.bias'}
                 or result.unexpected_keys):
        raise ValueError('Unexpected parent architecture or missing inherited parameter')
    return model


def run_balanced_weights(target):
    
    if target.ndim != 1 or not len(target) or not torch.all((target == 0) | (target == 1)):
        raise ValueError('Complete nonempty binary source target required')
    y = target.bool()
    starts = torch.ones(len(y), dtype=torch.bool, device=y.device)
    starts[1:] = y[1:] != y[:-1]
    ids = starts.long().cumsum(0) - 1
    lengths = torch.bincount(ids).float()
    classes = y[starts]
    both = bool(y.any() and (~y).any())
    class_mass = .5 if both else 1.
    numbers = torch.where(y, classes.sum(), (~classes).sum()).float()
    return class_mass / (lengths[ids] * numbers)


def followup_loss(prediction, frame, grid, run_balance):
    if run_balance:
        raw = F.binary_cross_entropy_with_logits(prediction['frame_logits'], frame.float(), reduction='none')
        state = (raw * run_balanced_weights(frame)).sum()
    else:
        state = balanced_bce(prediction['frame_logits'], frame)
    boundary = balanced_bce(prediction['boundary_logits'], boundary_targets(frame))
    region = balanced_bce(prediction['region_logits'], grid)
    total = state + .2 * boundary + .2 * region
    return total, {'state': float(state.detach()), 'boundary': float(boundary.detach()), 'region': float(region.detach())}
