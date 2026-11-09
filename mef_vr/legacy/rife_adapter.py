
from __future__ import annotations

import hashlib
import importlib.util
import math
import numbers
from pathlib import Path
import sys
import threading
import types
import uuid

import numpy as np
import torch
import torch.nn.functional as F

VERSION = "official_rife_m_native_auxiliary_v1"
DEFAULT_ASSET_ROOT = Path("/mnt/e/VRFlickerLab/models/rife_official_20261014_v1")
OFFICIAL_COMMIT = "5d8adbdd40e12c2c8f91930eff838aebe561c086"
CHECKPOINT_SHA256 = "9f9e2e8b5c3fef311c9a782aa17a30f87c388ce6dc9c00e4993eba3d3941d3cb"
SOURCE_SHA256 = {
    "model/IFNet_m.py": "e4e1e4d4f877ade9c8cdb1fd5b9e2f333670b5b6610d63b96e3a42a245c913d2",
    "model/refine.py": "547c92774aeca3a4a25ca0d061d01bb1115c84a7e475c40f0cc6ebfd759a7580",
    "model/warplayer.py": "eed94da2f2e8056fa0ceabed88b87fedf25ec849494991a956b9f2cbad33632c",
}
_IMPORT_LOCK = threading.Lock()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_assets(asset_root=DEFAULT_ASSET_ROOT):
    root = Path(asset_root).resolve()
    repo = root / "ECCV2022-RIFE"
    checkpoint = root / "RIFE_m_train_log/flownet.pkl"
    for relative, expected in SOURCE_SHA256.items():
        if sha256(repo / relative) != expected:
            raise ValueError(f"Official source hash mismatch: {relative}")
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Official checkpoint hash mismatch")
    receipt = root / "acquisition_receipt.json"
    return {
        "asset_root": str(root), "repository": str(repo),
        "official_commit": OFFICIAL_COMMIT,
        "source_sha256": dict(SOURCE_SHA256),
        "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
        "acquisition_receipt": str(receipt),
        "acquisition_receipt_sha256": sha256(receipt),
    }


def clean_state_dict(state):
    
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint must be a nonempty tensor state dictionary")
    result = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Non-tensor state entry")
        target = name.removeprefix("module.")
        if not target or target in result:
            raise ValueError("Ambiguous state-dictionary prefix conversion")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite checkpoint tensor")
        result[target] = value
    return result


def _load_audited_modules(repo, device):
    
    namespace = "_native_rife_" + uuid.uuid4().hex
    aliases = ("model", "model.warplayer", "model.refine", "model.IFNet_m")
    modules = {}
    with _IMPORT_LOCK:
        missing = object()
        previous = {name: sys.modules.get(name, missing) for name in aliases}
        package = types.ModuleType(namespace)
        package.__path__ = [str(repo / "model")]
        sys.modules[namespace] = package
        sys.modules["model"] = package
        try:
            for short in ("warplayer", "refine", "IFNet_m"):
                name = namespace + "." + short
                spec = importlib.util.spec_from_file_location(name, repo / "model" / (short + ".py"))
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                sys.modules["model." + short] = module
                setattr(package, short, module)
                spec.loader.exec_module(module)
                modules[short] = module
            modules["warplayer"].device = device
            modules["refine"].device = device
        finally:
            for name, old in previous.items():
                if old is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old
            for short in ("warplayer", "refine", "IFNet_m"):
                sys.modules.pop(namespace + "." + short, None)
            sys.modules.pop(namespace, None)
    return modules


def validate_pair(left, right, alpha):
    for value in (left, right):
        if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
            raise ValueError("Inputs must be uint8 RGB numpy arrays")
        if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
            raise ValueError("Inputs must be nonempty native HWC RGB")
    if left.shape != right.shape:
        raise ValueError("Left/right native shapes differ")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, numbers.Real):
        raise ValueError("Alpha must be a real interior timestamp fraction")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Alpha must be strictly between zero and one")
    return alpha


def explicit_device(device):
    result = torch.device(device)
    if result.type not in ("cpu", "cuda"):
        raise ValueError("Only explicit CPU or CUDA device is supported")
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda:0")
    return result


class RIFEAdapter:
    def __init__(self, *, device, asset_root=DEFAULT_ASSET_ROOT):
        self.device = explicit_device(device)
        self._assets = verified_assets(asset_root)
        
        state = torch.load(self._assets["checkpoint"], map_location="cpu", weights_only=True)
        state = clean_state_dict(state)
        self._modules = _load_audited_modules(Path(self._assets["repository"]), self.device)
        self._network = self._modules["IFNet_m"].IFNet_m()
        self._network.load_state_dict(state, strict=True)
        self._network.requires_grad_(False)
        self._network.eval()
        self._network.to(device=self.device, dtype=torch.float32)
        self._parameter_count = sum(value.numel() for value in self._network.parameters())
        self._state_keys = len(state)
        self._estimate_calls = 0

    @torch.inference_mode()
    def estimate(self, left_uint8_HWC, right_uint8_HWC, alpha):
        alpha = validate_pair(left_uint8_HWC, right_uint8_HWC, alpha)
        height, width = left_uint8_HWC.shape[:2]
        
        
        left = torch.from_numpy(left_uint8_HWC[:, :, ::-1].copy()).permute(2, 0, 1)[None]
        right = torch.from_numpy(right_uint8_HWC[:, :, ::-1].copy()).permute(2, 0, 1)[None]
        left = left.to(device=self.device, dtype=torch.float32) / 255.0
        right = right.to(device=self.device, dtype=torch.float32) / 255.0
        padding = (0, (-width) % 32, 0, (-height) % 32)
        left, right = F.pad(left, padding), F.pad(right, padding)
        
        with torch.autocast(device_type=self.device.type, enabled=False):
            result = self._network(torch.cat((left, right), dim=1), scale=[4, 2, 1], timestep=alpha)
        prediction = result[2][2]
        expected = (1, 3, height + padding[3], width + padding[1])
        if tuple(prediction.shape) != expected:
            raise RuntimeError("RIFE returned an unexpected native padded shape")
        prediction = prediction[0, :, :height, :width].permute(1, 2, 0).float().cpu().numpy()
        prediction = prediction[:, :, ::-1].copy()
        if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
            raise RuntimeError("RIFE output is nonfinite or outside [0,1]")
        self._estimate_calls += 1
        return prediction

    def asset_receipt(self):
        return {
            **self._assets, "adapter_version": VERSION, "adapter_sha256": sha256(__file__),
            "device": str(self.device), "precision": "float32", "autocast": False,
            "global_tf32_flags_unchanged_by_adapter": True,
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "weights_only": True,
            "strict_state_dict": True, "state_dict_key_count": self._state_keys,
            "parameter_count": self._parameter_count,
            "scale": 1.0, "scale_list": [4, 2, 1], "padding_multiple": 32,
            "input_output_color": "RGB", "official_internal_color": "BGR",
            "original_size_fps_pts_unchanged": True, "source_frames_mutated": False,
            "center_or_labels_provided": False, "teacher_input_channels": 0,
            "external_gpu_lock_required": True, "gpu_lock_managed_by_adapter": False,
            "estimate_calls": self._estimate_calls,
            "purpose": "auxiliary prediction residual only, not frame replacement or validated flicker detection",
        }

    def provenance(self):
        return self.asset_receipt()



RifeAdapter = RIFEAdapter
