
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from flicker_component_ablation import AblatedStateHead, canonical_names, mask_evidence, parameter_sha
from flicker_temporal_connectivity import same_frame_conv

ARMS = ('fixed_mix', 'learned_mix')
GATE_NAMES = ('temporal1', 'temporal4', 'state', 'boundary')


class ContextRoutingHead(AblatedStateHead):
    
    def __init__(self, mode, width=24):
        if mode not in ARMS: raise ValueError('Unknown context routing mode')
        self.mode = mode
        super().__init__(canonical_names(), 'no_raft', width=width)
        self.gates = nn.ModuleDict({
            'temporal1': nn.Conv3d(2*width, 1, 1),
            'temporal4': nn.Conv3d(2*width, 1, 1),
            'state': nn.Conv1d(2*self.state[0].out_channels, 1, 1),
            'boundary': nn.Conv1d(2*self.boundary[0].out_channels, 1, 1),
        })
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight); nn.init.zeros_(gate.bias)

    def mix(self, layer, features, name):
        current = same_frame_conv(layer, features)
        temporal = layer(features)
        observed = torch.cat((current, temporal-current), dim=1)
        learned = torch.sigmoid(self.gates[name](observed))
        amount = learned if self.mode == 'learned_mix' else learned*0 + .5
        return current + amount*(temporal-current), amount

    def forward(self, features, *, return_gate_maps=False):
        features = mask_evidence(features, self.removed_channels)
        if not len(features): raise ValueError('A nonempty complete source is required')
        x = features.permute(1, 0, 2, 3)[None]
        x = F.gelu(self.embed(x)); x = x + F.gelu(self.spatial(x))
        maps = {}
        delta, maps['temporal1'] = self.mix(self.temporal1, x, 'temporal1')
        x = x + F.gelu(delta)
        delta, maps['temporal4'] = self.mix(self.temporal4, x, 'temporal4')
        x = x + F.gelu(delta)
        flat = x.flatten(3)
        local = flat.topk(min(8, flat.shape[-1]), dim=-1).values.mean(-1)
        pooled = torch.cat((flat.mean(-1), local), dim=1)
        state, maps['state'] = self.mix(self.state[0], pooled, 'state')
        boundary, maps['boundary'] = self.mix(self.boundary[0], pooled, 'boundary')
        result = dict(frame_logits=self.state[2](F.gelu(state))[0, 0],
            boundary_logits=self.boundary[2](F.gelu(boundary))[0].transpose(0, 1),
            region_logits=self.region(x)[0, 0])
        return (result, maps) if return_gate_maps else result


def initialized_routing(seed, device='cpu', width=24):
    models = {}
    for arm in ARMS:
        torch.manual_seed(seed)
        models[arm] = ContextRoutingHead(arm, width=width).to(device)
    hashes = {arm: parameter_sha(model) for arm, model in models.items()}
    if len(set(hashes.values())) != 1: raise ValueError('Unmatched initial tensors')
    return models, hashes
