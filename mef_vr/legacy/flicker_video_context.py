
from __future__ import annotations

import torch

from flicker_context_routing import ContextRoutingHead
from flicker_component_ablation import canonical_names, parameter_sha

ARMS=('frame_context','video_context')


def video_context_features(features):
    if features.ndim!=4 or features.shape[1]!=51 or min(features.shape)<1:
        raise ValueError('One complete nonempty T,51,H,W source required')
    if canonical_names()[50]!='frozen_r3d_context': raise ValueError('Evidence schema changed')
    result=features.clone()
    
    
    result[:,50:51]=features[:,50:51].mean(dim=0,keepdim=True)
    return result


class VideoContextRoutingHead(ContextRoutingHead):
    def __init__(self,context_mode,width=24):
        if context_mode not in ARMS: raise ValueError('Unknown direct-context readout')
        self.context_mode=context_mode
        super().__init__('learned_mix',width=width)

    def forward(self,features,*,return_gate_maps=False):
        if self.context_mode=='video_context': features=video_context_features(features)
        return super().forward(features,return_gate_maps=return_gate_maps)


def initialized(seed,device='cpu'):
    models={}
    for arm in ARMS:
        torch.manual_seed(seed)
        models[arm]=VideoContextRoutingHead(arm).to(device)
    hashes={a:parameter_sha(m) for a,m in models.items()}
    if len(set(hashes.values()))!=1: raise ValueError('Matched initial tensors differ')
    return models,hashes
