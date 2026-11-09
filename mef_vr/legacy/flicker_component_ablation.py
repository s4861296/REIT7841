
from __future__ import annotations

import hashlib

import torch

from flicker_multimodel_core import DenseStateHead


ARMS = {
    'full': '完整时间融合模型',
    'no_dino': '时间输入去掉全部 DINO 语义证据',
    'no_raft': '时间输入去掉全部 RAFT 运动证据',
    'no_r3d_lineage': '时间输入去掉 R3D 分数及其派生 C Mask',
    'no_c_mask': '时间输入仅去掉旧 C Mask，保留 R3D 分数',
}


def canonical_names():
    names = [f'dino_projection_{i}' for i in range(16)]
    for lag in (1, 4, 8):
        for direction in (-1, 1):
            names += [f'dino_cosine_{direction * lag}', f'dino_available_{direction * lag}']
    names += ['rgb_mean_r', 'rgb_mean_g', 'rgb_mean_b', 'rgb_std_r', 'rgb_std_g', 'rgb_std_b',
              'raw_previous_mean', 'raw_previous_max', 'raw_next_mean', 'raw_next_max',
              'raft_aligned_mean', 'raft_support', 'raft_consistent', 'raft_available',
              'raft_aligned_max', 'frozen_C_mask_fraction']
    names += [f'rgb_patch_difference_{d * lag}' for lag in (1, 4, 8) for d in (-1, 1)]
    return names + ['frozen_r3d_context']


def removed_indices(names, arm):
    if list(names) != canonical_names() or arm not in ARMS:
        raise ValueError('Exact versioned 51-channel evidence schema and known arm required')
    if arm == 'full':
        return ()
    if arm in ('no_dino', 'no_raft'):
        prefix = 'dino_' if arm == 'no_dino' else 'raft_'
        return tuple(i for i, n in enumerate(names) if n.startswith(prefix))
    removed = {'frozen_C_mask_fraction'}
    if arm == 'no_r3d_lineage':
        removed.add('frozen_r3d_context')
    return tuple(i for i, n in enumerate(names) if n in removed)


def mask_evidence(features, indices):
    
    if features.ndim != 4 or features.shape[1] != 51:
        raise ValueError('One complete T,51,Hgrid,Wgrid observation required')
    if not indices:
        return features
    z = features.clone()
    z[:, list(indices)] = 0
    return z


class AblatedStateHead(DenseStateHead):
    def __init__(self, names, arm, width=24):
        self.removed_channels = removed_indices(names, arm)
        super().__init__(len(names), width=width)

    def forward(self, features):
        return super().forward(mask_evidence(features, self.removed_channels))


def parameter_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode()); digest.update(a.tobytes())
    return digest.hexdigest()


def initialized_models(names, seed, device='cpu', width=24):
    models = {}
    for arm in ARMS:
        torch.manual_seed(seed)
        models[arm] = AblatedStateHead(names, arm, width=width).to(device)
    hashes = {a: parameter_sha(m) for a, m in models.items()}
    if len(set(hashes.values())) != 1:
        raise ValueError('Within-seed trainable initialization must be identical')
    return models, hashes
