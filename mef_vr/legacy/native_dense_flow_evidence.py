
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def _rgb_pair(first, second):
    if (first.ndim != 4 or first.shape[1] != 3 or min(first.shape) < 1
            or first.dtype != torch.uint8 or second.dtype != torch.uint8
            or first.shape != second.shape or first.device != second.device):
        raise ValueError('Require same-device native uint8 RGB [N,3,H,W] pairs')


@dataclass(frozen=True)
class NativePad:
    height: int
    width: int
    bottom: int
    right: int

    def unpad(self, value):
        if value.shape[-2:] != (self.height + self.bottom, self.width + self.right):
            raise ValueError('Output dimensions differ from the recorded padded grid')
        return value[..., :self.height, :self.width]


@torch.no_grad()
def prepare_raft_pair(first, second):
    
    _rgb_pair(first, second)
    h, w = first.shape[-2:]
    ph, pw = max(128, math.ceil(h / 8) * 8), max(128, math.ceil(w / 8) * 8)
    spec = NativePad(h, w, ph - h, pw - w)
    values = []
    for rgb in (first, second):
        value = rgb.float() / 127.5 - 1.
        if spec.bottom or spec.right:
            value = F.pad(value, (0, spec.right, 0, spec.bottom), mode='replicate')
        values.append(value.contiguous())
    return values[0], values[1], spec


def raft_correlation_storage_bytes(height, width, batch=1):
    
    if any(type(v) is not int or v < 1 for v in (height, width, batch)):
        raise ValueError('Require positive integer dimensions and batch')
    h, w = max(128, math.ceil(height / 8) * 8) // 8, max(128, math.ceil(width / 8) * 8) // 8
    pixels = h * w
    elements = 0
    for _ in range(4):
        elements += batch * pixels * h * w
        h, w = h // 2, w // 2
    return elements * 4


def _check_flow(flow, rgb):
    if (flow.shape != (rgb.shape[0], 2, *rgb.shape[-2:])
            or flow.dtype != torch.float32 or flow.device != rgb.device
            or not torch.isfinite(flow).all()):
        raise ValueError('Require finite FP32 flow on the same native RGB grid/device')


def _sampling_grid(flow):
    n, _, h, w = flow.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=flow.device, dtype=flow.dtype),
                            torch.arange(w, device=flow.device, dtype=flow.dtype), indexing='ij')
    x, y = xx[None] + flow[:, 0], yy[None] + flow[:, 1]
    support = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    
    
    grid = torch.stack((2 * (x + .5) / w - 1, 2 * (y + .5) / h - 1), dim=-1)
    return grid, support


@torch.no_grad()
def native_flow_evidence(previous, current, current_to_previous, previous_to_current):
    
    _rgb_pair(previous, current)
    _check_flow(current_to_previous, current)
    _check_flow(previous_to_current, previous)
    grid, sampling_support = _sampling_grid(current_to_previous)
    prev, curr = previous.float() / 255., current.float() / 255.
    sampled = F.grid_sample(prev, grid, mode='bilinear', padding_mode='border', align_corners=False)
    forward_at_previous = F.grid_sample(previous_to_current, grid, mode='bilinear',
                                        padding_mode='border', align_corners=False)
    cycle = current_to_previous + forward_at_previous
    _, roundtrip_inside = _sampling_grid(cycle)
    raw = (curr - prev).abs().amax(dim=1)
    sampled_residual = (curr - sampled).abs().amax(dim=1)
    return {
        'raw_residual': raw,
        'sampled_residual': sampled_residual,
        'aligned_residual': torch.where(sampling_support, sampled_residual, raw),
        'sampling_support': sampling_support,
        'roundtrip_support': sampling_support & roundtrip_inside,
        'cycle_error_px': torch.linalg.vector_norm(cycle, dim=1),
        'motion_magnitude_px': torch.linalg.vector_norm(current_to_previous, dim=1),
    }
