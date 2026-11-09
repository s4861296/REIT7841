
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.models.optical_flow.raft import CorrBlock
from torchvision.models.optical_flow._utils import grid_sample


DEFAULT_CHUNK_BYTES = 256 * 1024**2


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


class InplaceCorrBlock(CorrBlock):
    

    def _compute_corr_volume(self, fmap1, fmap2):
        if fmap1.dtype != torch.float32 or fmap2.dtype != torch.float32:
            raise ValueError('Only original FP32 correlations are supported')
        batch_size, num_channels, h, w = fmap1.shape
        a = fmap1.view(batch_size, num_channels, h*w)
        b = fmap2.view(batch_size, num_channels, h*w)
        corr = torch.matmul(a.transpose(1, 2), b)
        corr = corr.view(batch_size, h, w, 1, h, w)
        return corr.div_(torch.sqrt(torch.tensor(num_channels)))


class RecomputedCorrBlock(torch.nn.Module):
    

    def __init__(self, *, num_levels=4, radius=4, max_chunk_bytes=DEFAULT_CHUNK_BYTES):
        super().__init__()
        if num_levels < 1 or radius < 0 or max_chunk_bytes < 4:
            raise ValueError('Invalid correlation geometry or chunk budget')
        self.num_levels = num_levels
        self.radius = radius
        self.out_channels = num_levels * (2*radius+1)**2
        self.max_chunk_bytes = max_chunk_bytes
        self._fmaps = None
        self.stats = {}

    @property
    def corr_pyramid(self):
        
        return []

    @corr_pyramid.setter
    def corr_pyramid(self, value):
        if not isinstance(value, list) or value:
            raise ValueError('Recomputed correlation retains no dense pyramid')
        self._fmaps = None

    def build_pyramid(self, fmap1, fmap2):
        if (fmap1.shape != fmap2.shape or fmap1.ndim != 4 or fmap1.shape[0] != 1
                or fmap1.dtype != torch.float32 or fmap2.dtype != torch.float32
                or fmap1.device != fmap2.device):
            raise ValueError('Exactly one original paired FP32 feature map required')
        minimum = 2 * 2**(self.num_levels-1)
        if min(fmap1.shape[-2:]) < minimum:
            raise ValueError('Feature maps too small for the original pyramid')
        self._fmaps = (fmap1, fmap2)
        h, w = fmap1.shape[-2:]
        rows = min(h*w, self.max_chunk_bytes // (4*h*w))
        if rows < 1:
            raise ValueError('Budget cannot hold one complete target-pixel row')
        
        if rows >= 128:
            rows = rows // 128 * 128
        self.stats = dict(source_pixels=h*w, target_pixels=h*w, source_chunk_rows=rows,
                          max_raw_correlation_bytes=rows*h*w*4, index_calls=0,
                          original_full_correlation_bytes=(h*w)**2*4)

    def index_pyramid(self, centroids_coords):
        if self._fmaps is None:
            raise ValueError('Build original feature maps before indexing')
        fmap1, fmap2 = self._fmaps
        batch_size, channels, h, w = fmap1.shape
        if (centroids_coords.shape != (batch_size, 2, h, w)
                or centroids_coords.dtype != torch.float32
                or centroids_coords.device != fmap1.device):
            raise ValueError('Original FP32 centroid geometry required')
        side = 2*self.radius+1
        di = torch.linspace(-self.radius, self.radius, side)
        dj = torch.linspace(-self.radius, self.radius, side)
        delta = torch.stack(torch.meshgrid(di, dj, indexing='ij'), dim=-1).to(centroids_coords.device)
        delta = delta.view(1, side, side, 2)
        centers = centroids_coords.permute(0, 2, 3, 1).reshape(h*w, 1, 1, 2)
        a = fmap1.view(batch_size, channels, h*w)
        b = fmap2.view(batch_size, channels, h*w)
        output = fmap1.new_empty((h*w, self.out_channels))
        chunk_rows = self.stats['source_chunk_rows']
        for start in range(0, h*w, chunk_rows):
            end = min(start+chunk_rows, h*w)
            volume = torch.matmul(a[:, :, start:end].transpose(1, 2), b)
            volume.div_(torch.sqrt(torch.tensor(channels)))
            volume = volume.reshape(end-start, 1, h, w)
            local_centers = centers[start:end]
            for level in range(self.num_levels):
                sampled = grid_sample(volume, local_centers+delta, align_corners=True, mode='bilinear')
                output[start:end, level*side*side:(level+1)*side*side] = sampled.reshape(end-start, -1)
                local_centers = local_centers / 2
                if level+1 < self.num_levels:
                    volume = F.avg_pool2d(volume, kernel_size=2, stride=2)
            del volume, sampled, local_centers
        self.stats['index_calls'] += 1
        return output.view(batch_size, h, w, self.out_channels).permute(0, 3, 1, 2).contiguous()


def replace_corr_block(model, mode, max_chunk_bytes=DEFAULT_CHUNK_BYTES):
    
    old = model.corr_block
    options = dict(num_levels=old.num_levels, radius=old.radius)
    if mode == 'inplace':
        block = InplaceCorrBlock(**options)
    elif mode == 'recompute':
        block = RecomputedCorrBlock(**options, max_chunk_bytes=max_chunk_bytes)
    else:
        raise ValueError('Explicit inplace or recompute mode required')
    model.corr_block = block
    return model


def install_raft_adapter(frozen_code_dir, *, mode, parity_receipt, max_chunk_bytes=DEFAULT_CHUNK_BYTES):
    
    receipt = json.loads(Path(parity_receipt).read_text())
    source = Path(__file__).resolve()
    raft_module = importlib.import_module('torchvision.models.optical_flow.raft')
    producer = importlib.import_module('build_native_motion_train_cache')
    expected_producer = Path(frozen_code_dir).resolve()/'build_native_motion_train_cache.py'
    if Path(producer.__file__).resolve() != expected_producer:
        raise ValueError('Producer must come from the explicitly frozen retry snapshot')
    if (receipt.get('passed') is not True or mode not in receipt.get('passed_modes', [])
            or receipt.get('adapter_sha256') != sha(source)
            or receipt.get('torchvision_raft_sha256') != sha(raft_module.__file__)
            or receipt.get('frozen_producer_sha256') != sha(expected_producer)
            or receipt.get('max_chunk_bytes') != max_chunk_bytes):
        raise ValueError('Measured GPU parity receipt missing or changed')
    if getattr(producer.raft_large, '_retry_adapter_installed', False):
        raise ValueError('RAFT adapter already installed; refuse stacked patches')
    original_factory = producer.raft_large

    def factory(*args, **kwargs):
        return replace_corr_block(original_factory(*args, **kwargs), mode, max_chunk_bytes)

    factory._retry_adapter_installed = True
    producer.raft_large = factory
    return dict(mode=mode, adapter_sha256=sha(source), parity_receipt_sha256=sha(parity_receipt),
                frozen_producer_sha256=sha(expected_producer), max_chunk_bytes=max_chunk_bytes,
                weights_changed=False, resolution_changed=False, fps_changed=False,
                precision='FP32', updates_per_direction=12)
