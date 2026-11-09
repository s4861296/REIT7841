


from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
from types import SimpleNamespace
from typing import Any

import numpy as np

import train_r3d18_flicker_detector as common


PROTOCOL = "r3d18_frozen_backbone_exact_frame_temporal_fpn_v3"
TEMPORAL_METRICS_VERSION = "native_frames_clean_events_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--backbone-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--head-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--pair-ranking-weight", type=float, default=0.25)
    parser.add_argument("--pair-ranking-margin-logit", type=float, default=0.5)
    parser.add_argument("--positive-frame-weight-cap", type=float, default=100.0)
    parser.add_argument("--initial-head", type=Path)
    parser.add_argument("--initial-head-sha256")
    parser.add_argument("--fixed-frame-threshold", type=float)
    parser.add_argument("--boundary-negative-weight", type=float, default=0.0)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--short-event-max-frames", type=int, default=3)
    parser.add_argument("--difference-context", choices=("window_zero", "source_previous"),
                        default="window_zero")
    parser.add_argument(
        "--feature-mode",
        choices=("appearance", "appearance_delta_mean_max"),
        default="appearance_delta_mean_max",
    )
    parser.add_argument(
        "--dense-difference-branch",
        choices=("off", "shallow_native"),
        default="shallow_native",
    )
    parser.add_argument("--train-pairs-per-root", type=int)
    parser.add_argument("--validation-pairs-per-root", type=int)
    parser.add_argument("--validation-on-training-pairs", action="store_true")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--skip-source-sha256", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def native_adjacent_difference(native_normalized_rgb, previous_normalized_rgb=None):
    
    import torch

    adjacent = (native_normalized_rgb[:, :, 1:] - native_normalized_rgb[:, :, :-1]).abs()
    if previous_normalized_rgb is None:
        first = torch.zeros_like(native_normalized_rgb[:, :, :1])
    else:
        if previous_normalized_rgb.shape != native_normalized_rgb[:, :, :1].shape:
            raise ValueError("previous RGB must be one aligned native-resolution frame")
        first = (native_normalized_rgb[:, :, :1] - previous_normalized_rgb).abs()
    return torch.cat((first, adjacent), dim=2)


def previous_source_frame(video, start, difference_context, device):
    
    if difference_context not in ("window_zero", "source_previous"):
        raise ValueError("unknown difference context")
    if not 0 <= start < len(video):
        raise ValueError("window start outside source video")
    if difference_context == "window_zero" or start == 0:
        return None
    return common.normalized_window(video, start - 1, 1, device)[0]


def following_source_frame(video,start,window,device):
    index=start+window
    if index>=len(video):return None
    return common.normalized_window(video,index,1,device)[0]


def create_temporal_head(
    channels: int,
    window_frames: int,
    feature_mode: str,
    dense_difference_branch: str,
):
    import torch

    class TemporalFPNHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.window_frames = window_frames
            multiplier = 1 if feature_mode == "appearance" else 3
            self.projections = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(value * multiplier, channels, 1)
                    for value in (64, 128, 256, 512)
                ]
            )
            self.dense_difference_branch = dense_difference_branch
            if dense_difference_branch == "shallow_native":
                
                
                
                self.difference_encoder = torch.nn.Sequential(
                    torch.nn.Conv3d(
                        3,
                        16,
                        kernel_size=(1, 7, 7),
                        stride=(1, 4, 4),
                        padding=(0, 3, 3),
                        bias=False,
                    ),
                    torch.nn.GroupNorm(4, 16),
                    torch.nn.GELU(),
                    torch.nn.Conv3d(
                        16,
                        32,
                        kernel_size=(1, 3, 3),
                        stride=(1, 4, 4),
                        padding=(0, 1, 1),
                        bias=False,
                    ),
                    torch.nn.GroupNorm(8, 32),
                    torch.nn.GELU(),
                )
                self.difference_projection = torch.nn.Sequential(
                    torch.nn.Conv1d(64, channels, 1),
                    torch.nn.GELU(),
                    torch.nn.Conv1d(channels, channels, 3, padding=1),
                )
                fusion_inputs = 5
            else:
                self.difference_encoder = None
                self.difference_projection = None
                fusion_inputs = 4
            self.fusion = torch.nn.Sequential(
                torch.nn.Conv1d(channels * fusion_inputs, channels * 2, 3, padding=1),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Conv1d(channels * 2, 1, 1),
            )

        def forward(self, features, native_normalized_rgb, previous_normalized_rgb=None):
            import torch
            import torch.nn.functional as functional

            projected = []
            for projection, feature in zip(self.projections, features):
                value = projection(feature)
                if value.shape[-1] != self.window_frames:
                    value = functional.interpolate(
                        value,
                        size=self.window_frames,
                        mode="linear",
                        align_corners=False,
                    )
                projected.append(value)
            if self.difference_encoder is not None:
                dense_difference = native_adjacent_difference(
                    native_normalized_rgb, previous_normalized_rgb
                )
                encoded = self.difference_encoder(dense_difference)
                dense_sequence = torch.cat(
                    (
                        encoded.mean(dim=(-1, -2)),
                        encoded.amax(dim=(-1, -2)),
                    ),
                    dim=1,
                )
                projected.append(self.difference_projection(dense_sequence))
            return self.fusion(torch.cat(projected, dim=1)).squeeze(1)

    return TemporalFPNHead().to("cuda:0")


def load_frozen_backbone(selection_path: Path):
    import torch

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    model_path = selection_path.resolve().parent / selection["selected_checkpoint"]
    actual_hash = common.sha256_file(model_path)
    if actual_hash != selection["selected_checkpoint_sha256"]:
        raise RuntimeError("frozen R3D backbone SHA256 mismatch")
    backbone = common.create_model(
        SimpleNamespace(
            no_pretrained=True,
            stem_spatial_stride=int(selection["stem_spatial_stride"]),
        )
    )
    state = torch.load(model_path, map_location="cuda:0", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone, selection, model_path


def temporal_features(backbone, tensor, feature_mode: str):
    

    import torch

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        value = backbone.stem(tensor)
        layer1 = backbone.layer1(value)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)
        results = []
        for feature in (layer1, layer2, layer3, layer4):
            appearance = feature.mean(dim=(-1, -2))
            if feature_mode == "appearance":
                results.append(appearance.detach())
                continue
            
            
            
            adjacent = (feature[:, :, 1:] - feature[:, :, :-1]).abs()
            zero = torch.zeros_like(appearance[:, :, :1])
            adjacent_mean = adjacent.mean(dim=(-1, -2))
            adjacent_max = adjacent.amax(dim=(-1, -2))
            change_mean = torch.maximum(
                torch.cat((zero, adjacent_mean), dim=2),
                torch.cat((adjacent_mean, zero), dim=2),
            )
            change_max = torch.maximum(
                torch.cat((zero, adjacent_max), dim=2),
                torch.cat((adjacent_max, zero), dim=2),
            )
            results.append(
                torch.cat((appearance, change_mean, change_max), dim=1).detach()
            )
        return tuple(results)


def padded_frame_slice(values: np.ndarray, start: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros(window, dtype=np.float32)
    valid = np.zeros(window, dtype=np.float32)
    count = min(window, len(values) - start)
    if count > 0:
        result[:count] = values[start : start + count]
        valid[:count] = 1.0
    return result, valid


def frame_coverage(frame_count: int, starts: list[int], window: int) -> np.ndarray:
    coverage = np.zeros(frame_count, dtype=np.float32)
    for start in starts:
        coverage[start : min(start + window, frame_count)] += 1
    if not np.all(coverage > 0):
        raise RuntimeError("window plan does not cover every source frame")
    return coverage


def normal_pair_consistency(logits, targets, contribution, negative_count):
    
    import torch
    import torch.nn.functional as functional
    if negative_count <= 0:
        return logits.sum() * 0.0
    if logits.shape != targets.shape or logits.shape != contribution.shape or logits.shape[0] != 2:
        raise ValueError('Expected paired [2,T] logits/targets/contribution')
    if bool((targets[1] > 0).any()):
        raise ValueError('Clean teacher cannot have positive targets')
    excess = functional.relu(logits[0] - logits[1].detach())
    penalty = functional.smooth_l1_loss(excess, torch.zeros_like(excess), reduction='none')
    return (penalty * (targets[0] == 0) * contribution[0]).sum() / negative_count


def train_pair(
    backbone,
    head,
    optimizer,
    pair: list[dict[str, Any]],
    videos: dict[str, np.ndarray],
    positive_weight: float,
    options: argparse.Namespace,
    extra_negative_video=None,
    extra_negative_weight: float = 0.5,
    auxiliary_boundary_weight: float = 0.0,
    normal_consistency_weight: float = 0.0,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    bug = next(row for row in pair if row["role"] == "bug")
    clean = next(row for row in pair if row["role"] == "clean_or_fixed")
    bug_targets = common.frame_targets(bug).astype(np.float32)
    clean_targets = common.frame_targets(clean).astype(np.float32)
    starts = common.window_starts(len(bug_targets), options.window_frames, options.window_stride)
    clean_starts = common.window_starts(
        len(clean_targets), options.window_frames, options.window_stride
    )
    if starts != clean_starts or len(bug_targets) != len(clean_targets) or clean_targets.any():
        raise ValueError(f"invalid exact-frame pair: {bug['pair_id']}")
    coverage = frame_coverage(len(bug_targets), starts, options.window_frames)
    boundary_mask = boundary_negative_mask(
        bug_targets, options.boundary_radius, options.short_event_max_frames
    )
    boundary_count = float(boundary_mask.sum())
    positive_frame_count = max(1.0, float(bug_targets.sum()))
    pair_positive_weight = min(
        options.positive_frame_weight_cap,
        (2 * len(bug_targets) - positive_frame_count) / positive_frame_count,
    )
    pair_positive_weight = max(positive_weight, pair_positive_weight)

    if auxiliary_boundary_weight < 0 or normal_consistency_weight < 0:
        raise ValueError('Auxiliary/consistency weights cannot be negative')
    if auxiliary_boundary_weight:
        from r3d_boundary_auxiliary import boundary_targets, auxiliary_boundary_loss
        encoded=[boundary_targets(t) for t in (bug_targets,clean_targets)]
        aux_targets=np.stack([t for t,v in encoded])
        aux_valid=np.stack([v for t,v in encoded])
        aux_positive=(aux_targets*aux_valid).sum(axis=(0,2))
        aux_count=aux_valid.sum(axis=(0,2))
        aux_weight=np.where(aux_positive>0,np.minimum(20,(aux_count-aux_positive)/np.maximum(aux_positive,1)),1)
        aux_denominators=torch.as_tensor(aux_count,device='cuda:0',dtype=torch.float32)
        aux_positive_weights=torch.as_tensor(aux_weight,device='cuda:0',dtype=torch.float32)

    head.train()
    optimizer.zero_grad(set_to_none=True)
    classification_total = ranking_total = boundary_total = 0.0
    auxiliary_total = consistency_total = 0.0
    normal_frame_count = int((bug_targets == 0).sum())
    for start in starts:
        bug_tensor, _ = common.normalized_window(
            videos[bug["sample_id"]], start, options.window_frames, "cuda:0"
        )
        clean_tensor, _ = common.normalized_window(
            videos[clean["sample_id"]], start, options.window_frames, "cuda:0"
        )
        inputs = torch.cat((bug_tensor, clean_tensor), dim=0)
        context = getattr(options, "difference_context", "window_zero")
        previous = None
        if context == "source_previous" and start > 0:
            previous = torch.cat([
                previous_source_frame(videos[row["sample_id"]], start, context, "cuda:0")
                for row in (bug, clean)
            ], dim=0)
        features = temporal_features(backbone, inputs, options.feature_mode)
        following=None
        if getattr(head,'requires_next_source_frame',False):
            contexts=[following_source_frame(videos[r['sample_id']],start,options.window_frames,'cuda:0') for r in (bug,clean)]
            if contexts[0] is not None:following=torch.cat(contexts,dim=0)
        bug_slice, valid = padded_frame_slice(
            bug_targets, start, options.window_frames
        )
        clean_slice, _ = padded_frame_slice(
            clean_targets, start, options.window_frames
        )
        coverage_slice, _ = padded_frame_slice(coverage, start, options.window_frames)
        boundary_slice, _ = padded_frame_slice(boundary_mask, start, options.window_frames)
        inverse_coverage = np.zeros_like(coverage_slice)
        inverse_coverage[valid > 0] = 1.0 / coverage_slice[valid > 0]
        targets = torch.from_numpy(np.stack((bug_slice, clean_slice))).to("cuda:0")
        contribution = torch.from_numpy(np.stack((inverse_coverage, inverse_coverage))).to(
            "cuda:0"
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = (head(features,inputs,previous,following) if getattr(head,'requires_next_source_frame',False)
                      else head(features, inputs, previous)).float()
            element = functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            class_weight = torch.where(
                targets > 0,
                torch.full_like(targets, pair_positive_weight),
                torch.ones_like(targets),
            )
            classification = (element * class_weight * contribution).sum() / (
                2 * len(bug_targets)
            )
            positive = targets[0] > 0
            if positive.any():
                ranking_values = functional.relu(
                    options.pair_ranking_margin_logit
                    - (logits[0, positive] - logits[1, positive])
                )
                ranking = (
                    ranking_values * contribution[0, positive]
                ).sum() / positive_frame_count
            else:
                ranking = logits.new_zeros(())
            boundary = logits.new_zeros(())
            if options.boundary_negative_weight > 0 and boundary_count:
                boundary_weights = torch.from_numpy(boundary_slice * inverse_coverage).to("cuda:0")
                boundary = (element[0] * boundary_weights).sum() / boundary_count
            loss = (classification + options.pair_ranking_weight * ranking
                    + options.boundary_negative_weight * boundary)
            if normal_consistency_weight:
                consistency = normal_pair_consistency(logits, targets, contribution, normal_frame_count)
                loss = loss + normal_consistency_weight * consistency
                consistency_total += float(consistency.detach().cpu())
            if auxiliary_boundary_weight:
                aux_slices=np.stack([[padded_frame_slice(channel,start,options.window_frames)[0]
                                      for channel in video] for video in aux_targets])
                valid_slices=np.stack([[padded_frame_slice(channel,start,options.window_frames)[0]
                                        for channel in video] for video in aux_valid])
                aux_loss=auxiliary_boundary_loss(head.boundary_logits,
                    torch.as_tensor(aux_slices,device='cuda:0'),torch.as_tensor(valid_slices,device='cuda:0'),
                    contribution,aux_positive_weights,aux_denominators)
                loss=loss+auxiliary_boundary_weight*aux_loss
                auxiliary_total+=float(aux_loss.detach().cpu())
        loss.backward()
        classification_total += float(classification.detach().cpu())
        ranking_total += float(ranking.detach().cpu())
        boundary_total += float(boundary.detach().cpu())
        del (
            bug_tensor,
            clean_tensor,
            inputs,
            features,
            targets,
            contribution,
            logits,
            loss,
            previous,
            following,
        )
    hard_negative_total = 0.0
    if extra_negative_video is not None:
        hard_negative_total = accumulate_hard_negative(
            backbone, head, extra_negative_video, options, extra_negative_weight
        )
    torch.nn.utils.clip_grad_norm_(head.parameters(), options.gradient_clip)
    optimizer.step()
    return {
        "classification_loss": classification_total,
        "ranking_loss": ranking_total,
        "boundary_negative_loss": boundary_total,
        "boundary_negative_frames": int(boundary_count),
        "exact_positive_frames": int(bug_targets.sum()),
        "pair_positive_weight": pair_positive_weight,
        "windows": len(starts),
        "hard_negative_loss": hard_negative_total,
        "auxiliary_boundary_loss": auxiliary_total,
        "normal_pair_consistency_loss": consistency_total,
        "normal_pair_consistency_frames": normal_frame_count,
    }


def negative_window_weights(frame_count, starts, window):
    
    coverage = frame_coverage(frame_count, starts, window)
    result = []
    for start in starts:
        segment, valid = padded_frame_slice(coverage, start, window)
        weight = np.zeros(window, dtype=np.float32)
        weight[valid > 0] = 1.0 / segment[valid > 0] / frame_count
        result.append(weight)
    return result


def accumulate_hard_negative(backbone, head, video, options, weight):
    
    import torch
    import torch.nn.functional as functional
    if weight <= 0:
        raise ValueError('Hard negative weight must be positive')
    starts = common.window_starts(len(video), options.window_frames, options.window_stride)
    weights = negative_window_weights(len(video), starts, options.window_frames)
    total = 0.0
    for start, contribution in zip(starts, weights):
        inputs, _ = common.normalized_window(video, start, options.window_frames, 'cuda:0')
        previous = previous_source_frame(video, start, getattr(options, 'difference_context', 'window_zero'), 'cuda:0')
        features = temporal_features(backbone, inputs, options.feature_mode)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = head(features, inputs, previous).float()
            loss = weight * (functional.binary_cross_entropy_with_logits(
                logits, torch.zeros_like(logits), reduction='none') *
                torch.from_numpy(contribution).to('cuda:0')).sum()
        loss.backward()
        total += float(loss.detach().cpu())
        del inputs, previous, features, logits, loss
    return total


def infer_record(backbone, head, record: dict[str, Any], video: np.ndarray, options):
    import torch

    frame_count = len(video)
    starts = common.window_starts(frame_count, options.window_frames, options.window_stride)
    probability_sum = np.zeros(frame_count, dtype=np.float64)
    probability_count = np.zeros(frame_count, dtype=np.int32)
    boundary_sum=np.zeros((2,frame_count),dtype=np.float64) if hasattr(head,'boundary_logits') else None
    head.eval()
    with torch.inference_mode():
        for start in starts:
            tensor, _ = common.normalized_window(
                video, start, options.window_frames, "cuda:0"
            )
            features = temporal_features(backbone, tensor, options.feature_mode)
            previous = previous_source_frame(
                video, start, getattr(options, "difference_context", "window_zero"), "cuda:0"
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                following=(following_source_frame(video,start,options.window_frames,'cuda:0')
                           if getattr(head,'requires_next_source_frame',False) else None)
                raw_logits=(head(features,tensor,previous,following) if getattr(head,'requires_next_source_frame',False)
                            else head(features,tensor,previous))
                probabilities = torch.sigmoid(
                    raw_logits.float()
                )[0].cpu().numpy()
            count = min(options.window_frames, frame_count - start)
            probability_sum[start : start + count] += probabilities[:count]
            probability_count[start : start + count] += 1
            if boundary_sum is not None:
                boundary_sum[:,start:start+count]+=torch.sigmoid(head.boundary_logits.float())[0,:,:count].cpu().numpy()
            del tensor, features, previous, following, raw_logits
    if not np.all(probability_count > 0):
        raise RuntimeError(f"inference left frames uncovered: {record['sample_id']}")
    frame_probabilities = probability_sum / probability_count
    targets = common.frame_targets(record).astype(np.uint8)
    result = {
        "sample_id": record["sample_id"],
        "pair_id": record["pair_id"],
        "role": record["role"],
        "root_cause": common.root_of(record),
        "camera_condition": record["camera_condition"],
        "fps": float(record["fps"]),
        "frame_count": frame_count,
        "frame_probabilities": frame_probabilities.tolist(),
        "frame_targets": targets.tolist(),
        "all_source_frames_covered": True,
        "exact_frame_supervision": True,
        "difference_context": getattr(options, "difference_context", "window_zero"),
    }
    if boundary_sum is not None:
        result['onset_probabilities']=(boundary_sum[0]/probability_count).tolist()
        result['last_abnormal_frame_probabilities']=(boundary_sum[1]/probability_count).tolist()
        result['auxiliary_boundaries_used_to_change_primary_predictions']=False
    return result


def frame_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = scores >= threshold
    expected = labels.astype(bool)
    tp = int(np.logical_and(prediction, expected).sum())
    fp = int(np.logical_and(prediction, ~expected).sum())
    fn = int(np.logical_and(~prediction, expected).sum())
    tn = int(np.logical_and(~prediction, ~expected).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {
        "threshold": threshold,
        "tp_frames": tp,
        "fp_frames": fp,
        "fn_frames": fn,
        "tn_frames": tn,
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "iou": iou,
    }


def boundary_negative_mask(targets: np.ndarray, radius: int = 2,
                           max_event_frames: int = 3) -> np.ndarray:
    
    if radius < 0 or max_event_frames < 1:
        raise ValueError("invalid boundary neighbourhood")
    expected = np.asarray(targets, dtype=bool)
    selected = np.zeros(len(expected), dtype=bool)
    for start, end in intervals(expected):
        if end - start + 1 <= max_event_frames:
            selected[max(0, start-radius): min(len(expected), end+radius+1)] = True
    return (selected & ~expected).astype(np.float32)


def calibrate_frame_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.concatenate([np.asarray(row["frame_targets"], dtype=np.uint8) for row in rows])
    scores = np.concatenate(
        [np.asarray(row["frame_probabilities"], dtype=np.float64) for row in rows]
    )
    candidates = np.linspace(0.005, 0.995, 199)
    metrics = [frame_metrics(labels, scores, float(threshold)) for threshold in candidates]
    return max(metrics, key=lambda item: (item["f1"], item["iou"], item["specificity"]))


def intervals(values: np.ndarray) -> list[tuple[int, int]]:
    active = np.flatnonzero(values).tolist()
    if not active:
        return []
    result = []
    start = previous = active[0]
    for frame in active[1:]:
        if frame != previous + 1:
            result.append((start, previous))
            start = frame
        previous = frame
    result.append((start, previous))
    return result


def interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    union = max(left[1], right[1]) - min(left[0], right[0]) + 1
    return intersection / union


def detailed_temporal_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    positive_ious = []
    event_tp = event_fp = event_fn = 0
    onset_error_seconds = []
    offset_error_seconds = []
    false_positive_frames = 0
    clean_false_events = 0
    clean_videos = clean_false_videos = 0
    clean_minutes = 0.0
    for row in rows:
        expected = np.asarray(row["frame_targets"], dtype=bool)
        predicted = np.asarray(row["frame_probabilities"]) >= threshold
        if expected.ndim != 1 or expected.shape != predicted.shape or not len(expected):
            raise ValueError("temporal metrics require aligned, nonempty native-frame arrays")
        if not np.isfinite(row["fps"]) or row["fps"] <= 0:
            raise ValueError("FPS must be finite and positive")
        if expected.any():
            union = np.logical_or(expected, predicted).sum()
            positive_ious.append(
                float(np.logical_and(expected, predicted).sum() / union) if union else 1.0
            )
            expected_events = intervals(expected)
            predicted_events = intervals(predicted)
            unused = set(range(len(predicted_events)))
            for target in expected_events:
                choices = sorted(
                    ((interval_iou(target, predicted_events[index]), index) for index in unused),
                    reverse=True,
                )
                if choices and choices[0][0] >= 0.5:
                    _, index = choices[0]
                    unused.remove(index)
                    event_tp += 1
                    onset_error_seconds.append(
                        abs(predicted_events[index][0] - target[0]) / row["fps"]
                    )
                    offset_error_seconds.append(
                        abs(predicted_events[index][1] - target[1]) / row["fps"]
                    )
                else:
                    event_fn += 1
            event_fp += len(unused)
        else:
            
            count = len(intervals(predicted))
            event_fp += count
            clean_false_events += count
            clean_videos += 1
            clean_false_videos += int(count > 0)
            false_positive_frames += int(predicted.sum())
            clean_minutes += len(predicted) / row["fps"] / 60
    event_precision = event_tp / (event_tp + event_fp) if event_tp + event_fp else 0.0
    event_recall = event_tp / (event_tp + event_fn) if event_tp + event_fn else 0.0
    return {
        "metrics_version": TEMPORAL_METRICS_VERSION,
        "boundary_error_population": "matched_events_only",
        "event_matching": "chronological_gt_greedy_max_iou_one_to_one",
        "mean_exact_frame_iou_positive_videos": (
            statistics.fmean(positive_ious) if positive_ious else None
        ),
        "event_match_iou_threshold": 0.5,
        "event_tp": event_tp,
        "event_fp": event_fp,
        "event_fn": event_fn,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": (
            2 * event_precision * event_recall / (event_precision + event_recall)
            if event_precision + event_recall
            else 0.0
        ),
        "mean_onset_absolute_error_seconds": (
            statistics.fmean(onset_error_seconds) if onset_error_seconds else None
        ),
        "mean_offset_absolute_error_seconds": (
            statistics.fmean(offset_error_seconds) if offset_error_seconds else None
        ),
        "false_positive_frames_per_clean_minute": (
            false_positive_frames / clean_minutes if clean_minutes else None
        ),
        "clean_video_count": clean_videos,
        "clean_videos_with_any_frame_fp": clean_false_videos,
        "clean_false_positive_frames": false_positive_frames,
        "clean_false_positive_events": clean_false_events,
        "clean_duration_minutes": clean_minutes,
        "false_positive_events_per_clean_minute": (
            clean_false_events / clean_minutes if clean_minutes else None
        ),
        "clean_video_specificity_any_frame_rule": (
            (clean_videos - clean_false_videos) / clean_videos if clean_videos else None
        ),
        "temporal_resolution": "native_source_frame",
    }


def evaluate(backbone, head, pairs, options):
    rows = []
    for pair_index, pair in enumerate(pairs, start=1):
        for record in pair:
            video, audit = common.decode_video(record)
            row = infer_record(backbone, head, record, video, options)
            row["input_audit"] = audit
            rows.append(row)
            del video
        print(
            common.canonical_json(
                {"frame_validation_progress": f"{pair_index}/{len(pairs)}"}
            ),
            flush=True,
        )
    if options.fixed_frame_threshold is None:
        metric = calibrate_frame_threshold(rows)
    else:
        metric = frame_metrics(
            np.concatenate([row["frame_targets"] for row in rows]),
            np.concatenate([row["frame_probabilities"] for row in rows]),
            options.fixed_frame_threshold,
        )
    metric["threshold_recalibrated"] = options.fixed_frame_threshold is None
    metric["temporal"] = detailed_temporal_metrics(rows, metric["threshold"])
    by_root = {}
    for root in sorted({row["root_cause"] for row in rows}):
        subset = [row for row in rows if row["root_cause"] == root]
        labels = np.concatenate([np.asarray(row["frame_targets"]) for row in subset])
        scores = np.concatenate([np.asarray(row["frame_probabilities"]) for row in subset])
        by_root[root] = frame_metrics(labels, scores, metric["threshold"])
    metric["by_root_cause"] = by_root
    return rows, metric


def main() -> None:
    options = parse_args()
    if options.window_frames != 32:
        raise SystemExit("the current multiscale head requires 32-frame windows")
    if options.window_stride > options.window_frames:
        raise SystemExit("window stride cannot exceed window length")
    if options.boundary_negative_weight < 0 or options.boundary_radius < 0:
        raise SystemExit("boundary weight/radius must be nonnegative")
    if options.short_event_max_frames < 1:
        raise SystemExit("short event maximum must be positive")
    if options.difference_context == "source_previous" and options.dense_difference_branch != "shallow_native":
        raise SystemExit("source_previous requires shallow_native difference branch")
    if options.fixed_frame_threshold is not None and not 0 <= options.fixed_frame_threshold <= 1:
        raise SystemExit("threshold out of range")
    if bool(options.initial_head) != bool(options.initial_head_sha256):
        raise SystemExit("initial head path and SHA256 must be provided together")
    if options.output_dir.exists() and any(options.output_dir.iterdir()):
        raise SystemExit("refusing to overwrite a nonempty training output directory")
    import torch

    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    backbone, selection, backbone_path = load_frozen_backbone(
        options.backbone_selection
    )
    train_records = common.read_jsonl(options.train_manifest)
    validation_records = common.read_jsonl(options.validation_manifest)
    train_pairs = common.select_pairs(
        train_records, options.train_pairs_per_root, options.seed
    )
    validation_pairs = (
        train_pairs
        if options.validation_on_training_pairs
        else common.select_pairs(
            validation_records, options.validation_pairs_per_root, options.seed + 1
        )
    )
    if not options.validation_on_training_pairs:
        train_scenes = {row["scene_id"] for pair in train_pairs for row in pair}
        validation_scenes = {
            row["scene_id"] for pair in validation_pairs for row in pair
        }
        if train_scenes & validation_scenes:
            raise RuntimeError("scene leakage between frame train and validation")

    positive_frames = sum(
        int(common.frame_targets(row).sum())
        for pair in train_pairs
        for row in pair
    )
    total_frames = sum(int(row["frame_count"]) for pair in train_pairs for row in pair)
    negative_frames = total_frames - positive_frames
    positive_weight = min(
        options.positive_frame_weight_cap,
        negative_frames / max(1, positive_frames),
    )
    head = create_temporal_head(
        options.head_channels,
        options.window_frames,
        options.feature_mode,
        options.dense_difference_branch,
    )
    if options.initial_head:
        if common.sha256_file(options.initial_head) != options.initial_head_sha256:
            raise RuntimeError("initial head SHA256 mismatch")
        head.load_state_dict(torch.load(options.initial_head, map_location="cuda:0", weights_only=True), strict=True)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    protocol = {
        "schema_version": "1.0.0",
        "protocol": PROTOCOL,
        "created_utc": utc_now(),
        "train_manifest": str(options.train_manifest.resolve()),
        "train_manifest_sha256": common.sha256_file(options.train_manifest),
        "validation_manifest": str(options.validation_manifest.resolve()),
        "validation_manifest_sha256": common.sha256_file(options.validation_manifest),
        "backbone_selection": str(options.backbone_selection.resolve()),
        "backbone_selection_sha256": common.sha256_file(options.backbone_selection),
        "backbone_model": str(backbone_path),
        "backbone_model_sha256": selection["selected_checkpoint_sha256"],
        "backbone_frozen": True,
        "initial_head": str(options.initial_head) if options.initial_head else None,
        "initial_head_sha256": options.initial_head_sha256,
        "optimizer_reset": True,
        "fixed_frame_threshold": options.fixed_frame_threshold,
        "boundary_negative_weight": options.boundary_negative_weight,
        "boundary_radius": options.boundary_radius,
        "short_event_max_frames": options.short_event_max_frames,
        "difference_context": options.difference_context,
        "boundary_loss_definition": "bug_negative_neighbours_mean_BCE_inverse_window_coverage",
        "epochs": options.epochs,
        "learning_rate": options.learning_rate,
        "weight_decay": options.weight_decay,
        "train_pair_ids_in_order": [pair[0]["pair_id"] for pair in train_pairs],
        "input_native_resolution": True,
        "input_all_source_frames": True,
        "input_spatial_resize": False,
        "input_temporal_sampling": False,
        "window_frames": options.window_frames,
        "window_stride": options.window_stride,
        "feature_temporal_lengths": [32, 16, 8, 4],
        "feature_mode": options.feature_mode,
        "temporal_change_statistics": (
            [
                "adjacent_feature_absolute_difference_spatial_mean",
                "adjacent_feature_absolute_difference_spatial_max",
            ]
            if options.feature_mode == "appearance_delta_mean_max"
            else []
        ),
        "dense_difference_branch": options.dense_difference_branch,
        "dense_difference_details": (
            {
                "source": "absolute difference of adjacent native RGB frames",
                "temporal_stride": 1,
                "temporal_output_length": options.window_frames,
                "learned_spatial_strides": [4, 4],
                "spatial_statistics": ["mean", "maximum"],
                "motion_compensation": False,
                "note": "Hard-negative and motion-compensation stages are deferred.",
            }
            if options.dense_difference_branch == "shallow_native"
            else None
        ),
        "output": "one probability per native source frame",
        "frame_label_source": "exact answer.events start_frame/end_frame",
        "overlap_loss_weighting": "inverse source-frame window coverage",
        "positive_frame_weight": positive_weight,
        "pair_ranking_weight": options.pair_ranking_weight,
        "pair_ranking_margin_logit": options.pair_ranking_margin_logit,
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "validation_on_training_pairs": options.validation_on_training_pairs,
        "test_used": False,
        "real_gt_used": False,
        "seed": options.seed,
    }
    (options.output_dir / "training_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    history = []
    for epoch in range(1, options.epochs + 1):
        epoch_classification = []
        epoch_ranking = []
        for index, pair in enumerate(train_pairs, start=1):
            videos = {}
            for record in pair:
                if not options.skip_source_sha256 and record.get("video_sha256"):
                    if common.sha256_file(Path(record["video_path"])) != record["video_sha256"]:
                        raise RuntimeError(f"source hash mismatch: {record['sample_id']}")
                video, _ = common.decode_video(record)
                videos[record["sample_id"]] = video
            outcome = train_pair(
                backbone,
                head,
                optimizer,
                pair,
                videos,
                positive_weight,
                options,
            )
            epoch_classification.append(outcome["classification_loss"])
            epoch_ranking.append(outcome["ranking_loss"])
            print(
                common.canonical_json(
                    {
                        "epoch": epoch,
                        "frame_pair_progress": f"{index}/{len(train_pairs)}",
                        "pair_id": pair[0]["pair_id"],
                        "classification_loss": round(
                            outcome["classification_loss"], 6
                        ),
                        "ranking_loss": round(outcome["ranking_loss"], 6),
                        "boundary_negative_loss": round(outcome["boundary_negative_loss"], 6),
                        "boundary_negative_frames": outcome["boundary_negative_frames"],
                        "exact_positive_frames": outcome["exact_positive_frames"],
                        "pair_positive_weight": round(
                            outcome["pair_positive_weight"], 6
                        ),
                        "windows_per_video": outcome["windows"],
                    }
                ),
                flush=True,
            )
            del videos
        predictions, validation = evaluate(
            backbone, head, validation_pairs, options
        )
        result = {
            "epoch": epoch,
            "mean_training_classification_loss": statistics.fmean(
                epoch_classification
            ),
            "mean_training_pair_ranking_loss": statistics.fmean(epoch_ranking),
            "validation": validation,
        }
        history.append(result)
        checkpoint = options.output_dir / f"checkpoint-{epoch:04d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), checkpoint / "temporal_head.pt")
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        (checkpoint / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (checkpoint / "validation_frame_predictions.jsonl").write_text(
            "".join(common.canonical_json(row) + "\n" for row in predictions),
            encoding="utf-8",
        )
        print(
            common.canonical_json(
                {"frame_epoch_complete": epoch, "validation": validation}
            ),
            flush=True,
        )
    (options.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "protocol": PROTOCOL,
                "completed_utc": utc_now(),
                "status": "completed",
                "history": history,
                "test_used": False,
                "real_gt_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
