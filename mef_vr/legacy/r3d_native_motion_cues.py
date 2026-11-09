
import torch
import torch.nn.functional as F
from r3d_native_restoration_aux import NativeRestorationRefiner


CUE_NAMES = ('aligned_residual', 'sampling_support', 'consistent', 'has_previous')


def motion_cues(aligned_residual, sampling_support, consistent, has_previous):
    
    x = aligned_residual
    if (x.ndim != 3 or min(x.shape) < 1 or x.dtype != torch.float32
            or not torch.isfinite(x).all() or (x < 0).any() or (x > 1).any()):
        raise ValueError('Finite original-grid FP32 normalized residual required')
    for flag in (sampling_support, consistent):
        if flag.dtype != torch.bool or flag.shape != x.shape or flag.device != x.device:
            raise ValueError('Native boolean correspondence maps required')
    if (has_previous.dtype != torch.bool or has_previous.shape != (x.shape[0],)
            or has_previous.device != x.device):
        raise ValueError('Original-video predecessor flags required')
    if (consistent & ~sampling_support).any():
        raise ValueError('Consistency cannot be asserted outside sampling support')
    missing = ~has_previous
    if (x[missing] != 0).any() or sampling_support[missing].any() or consistent[missing].any():
        raise ValueError('Missing predecessor must retain explicit empty evidence')
    hp = has_previous[:, None, None].expand_as(x)
    return torch.stack((x * sampling_support, sampling_support, consistent, hp), dim=1).detach()


class NativeMotionCueRefiner(NativeRestorationRefiner):
    
    def __init__(self, use_native_skips=False):
        super().__init__(use_native_skips)
        self.motion_projection = torch.nn.Conv2d(len(CUE_NAMES), 8, 1, bias=False)
        torch.nn.init.zeros_(self.motion_projection.weight)
        for parameter in self.restoration_head.parameters():
            parameter.requires_grad_(False)

    def load_parent(self, state):
        expected = set(self.state_dict()) - {'motion_projection.weight'}
        if set(state) != expected:
            raise ValueError('Complete unextended restoration parent state required')
        merged = dict(state)
        merged['motion_projection.weight'] = torch.zeros_like(self.motion_projection.weight)
        self.load_state_dict(merged, strict=True)

    def decode_with_motion(self, features, cues):
        if features.ndim != 5 or features.shape[2] != 22 or min(features.shape) < 1:
            raise ValueError('Complete B,T,22,H,W native features required')
        b, t, c, h, w = features.shape
        if (cues.shape != (b, t, len(CUE_NAMES), h, w)
                or cues.dtype != features.dtype or cues.device != features.device
                or not torch.isfinite(cues).all() or (cues < 0).any() or (cues > 1).any()):
            raise ValueError('Same-coordinate finite normalized motion cues required')
        
        
        fine = self.stem(features.reshape(b*t, c, h, w))
        fine = fine + self.motion_projection(cues.reshape(b*t, len(CUE_NAMES), h, w))
        mid = self.down2(fine)
        coarse = self.down4(mid)
        ch, cw = coarse.shape[-2:]
        sequence = coarse.reshape(b, t, 32, ch, cw).permute(0, 2, 1, 3, 4)
        sequence = sequence + self.temporal(sequence)
        coarse = sequence.permute(0, 2, 1, 3, 4).reshape(b*t, 32, ch, cw)
        up = F.interpolate(coarse, size=mid.shape[-2:], mode='bilinear', align_corners=False)
        up = self.up2(torch.cat((up, mid if self.use_native_skips else torch.zeros_like(mid)), dim=1))
        up = F.interpolate(up, size=(h, w), mode='bilinear', align_corners=False)
        return self.up1(torch.cat((up, fine if self.use_native_skips else torch.zeros_like(fine)), dim=1))

    def forward(self, features, cues):
        b, t, _, h, w = features.shape
        return self.classifier(self.decode_with_motion(features, cues)).reshape(b, t, h, w)

    def forward_with_restoration(self, *args, **kwargs):
        raise NotImplementedError('Motion candidate uses pixel BCE, not restoration auxiliary loss')
