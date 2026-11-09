
from __future__ import annotations

import torch
from torch.nn import functional as F

from flicker_component_ablation import AblatedStateHead, canonical_names, mask_evidence, parameter_sha

ARMS = ('dense_time', 'same_frame')


def same_frame_conv(layer, x):
    
    if isinstance(layer, torch.nn.Conv3d):
        if layer.kernel_size[1:] != (1, 1) or layer.stride != (1, 1, 1) or layer.groups != 1:
            raise ValueError('Only existing time-only Conv3d layers are supported')
        return F.conv3d(x, layer.weight.sum(2, keepdim=True), layer.bias)
    if isinstance(layer, torch.nn.Conv1d):
        if layer.stride != (1,) or layer.groups != 1:
            raise ValueError('Only existing stride1 temporal readout is supported')
        return F.conv1d(x, layer.weight.sum(2, keepdim=True), layer.bias)
    raise ValueError('Unexpected learned temporal layer')


class ConnectivityHead(AblatedStateHead):
    def __init__(self, mode, width=24):
        if mode not in ARMS: raise ValueError('Unknown connectivity mode')
        self.mode = mode
        super().__init__(canonical_names(), 'no_raft', width=width)

    def forward(self, features):
        if self.mode == 'dense_time': return super().forward(features)
        features = mask_evidence(features, self.removed_channels)
        x = features.permute(1, 0, 2, 3)[None]
        x = F.gelu(self.embed(x))
        x = x + F.gelu(self.spatial(x))
        x = x + F.gelu(same_frame_conv(self.temporal1, x))
        x = x + F.gelu(same_frame_conv(self.temporal4, x))
        flat = x.flatten(3)
        local = flat.topk(min(8, flat.shape[-1]), dim=-1).values.mean(-1)
        pooled = torch.cat((flat.mean(-1), local), dim=1)
        state = self.state[2](self.state[1](same_frame_conv(self.state[0], pooled)))
        boundary = self.boundary[2](self.boundary[1](same_frame_conv(self.boundary[0], pooled)))
        return dict(frame_logits=state[0, 0], boundary_logits=boundary[0].transpose(0, 1),
                    region_logits=self.region(x)[0, 0])


def initialized_connectivity(seed, device='cpu', width=24):
    result = {}
    for arm in ARMS:
        torch.manual_seed(seed)
        result[arm] = ConnectivityHead(arm, width=width).to(device)
    hashes = {a: parameter_sha(m) for a, m in result.items()}
    if len(set(hashes.values())) != 1: raise ValueError('Unmatched parameter initialization')
    return result, hashes
