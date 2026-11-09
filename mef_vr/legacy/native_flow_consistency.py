
import math
import torch
import torch.nn.functional as F
from native_dense_flow_evidence import _sampling_grid

RELATIVE_SQUARED_THRESHOLD=.01
ABSOLUTE_SQUARED_THRESHOLD=.5


@torch.no_grad()
def bidirectional_consistency(current_to_previous,previous_to_current,*,
                              alpha=RELATIVE_SQUARED_THRESHOLD,beta=ABSOLUTE_SQUARED_THRESHOLD):
    
    a,b=current_to_previous,previous_to_current
    if (a.ndim!=4 or a.shape[1]!=2 or min(a.shape)<1 or a.dtype!=torch.float32
            or b.shape!=a.shape or b.dtype!=torch.float32 or b.device!=a.device
            or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
        raise ValueError('Finite same-device FP32 N,2,H,W native flows required')
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in (alpha,beta)) or alpha<0 or beta<=0:
        raise ValueError('Finite alpha>=0 and beta>0 required')
    grid,support=_sampling_grid(a)
    reverse=F.grid_sample(b,grid,mode='bilinear',padding_mode='border',align_corners=False)
    error_squared=(a+reverse).square().sum(1)
    threshold=alpha*(a.square().sum(1)+reverse.square().sum(1))+beta
    if not torch.isfinite(error_squared).all() or not torch.isfinite(threshold).all():
        raise ValueError('Squared displacement overflow')
    consistent=support & (error_squared<threshold)
    inconsistent=support & ~consistent
    return dict(sampling_support=support,consistent=consistent,inconsistent_within_bounds=inconsistent,
                cycle_error_squared_px2=error_squared,threshold_squared_px2=threshold)
