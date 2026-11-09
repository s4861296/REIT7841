
import numpy as np


def boundary_targets(labels):
    
    labels=np.asarray(labels)
    if labels.ndim!=1 or not len(labels) or not np.isin(labels,[0,1]).all():
        raise ValueError('Require a nonempty binary source-frame sequence')
    active=labels.astype(bool)
    targets=np.zeros((2,len(labels)),dtype=np.float32)
    targets[0,1:]=active[1:] & ~active[:-1]
    targets[1,:-1]=active[:-1] & ~active[1:]
    valid=np.ones_like(targets);valid[0,0]=0;valid[1,-1]=0
    return targets,valid


def wrap_boundary_head(base):
    
    import torch

    class BoundaryHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base=base
            last=base.fusion[-1]
            self.boundary_projection=torch.nn.Conv1d(last.in_channels,2,1).to(last.weight.device)
            self.boundary_logits=None
            self._hidden=None
            self.base.fusion[-2].register_forward_hook(self._capture)

        def _capture(self,module,args,output):
            self._hidden=output

        def forward(self,features,rgb,previous=None):
            primary=self.base(features,rgb,previous)
            self.boundary_logits=self.boundary_projection(self._hidden)
            self._hidden=None
            return primary

    return BoundaryHead()


def auxiliary_boundary_loss(logits,targets,valid,contribution,positive_weights,denominators):
    
    import torch
    import torch.nn.functional as F
    element=F.binary_cross_entropy_with_logits(logits.float(),targets,reduction='none')
    weights=torch.where(targets>0,positive_weights[None,:,None],torch.ones_like(targets))
    numerator=(element*weights*valid*contribution[:,None,:]).sum(dim=(0,2))
    return (numerator/denominators.clamp_min(1)).mean()
