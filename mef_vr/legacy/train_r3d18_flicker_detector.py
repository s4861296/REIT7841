


from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import numpy as np


PROTOCOL = "r3d18_native_full_video_flicker_v1"
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=("global", "oracle_roi"), default="global")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--stem-spatial-stride", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--pair-ranking-weight", type=float, default=0.25)
    parser.add_argument("--pair-ranking-margin-logit", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--roi-margin-ratio", type=float, default=0.15)
    parser.add_argument("--roi-min-side", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--train-pairs-per-root", type=int)
    parser.add_argument("--validation-pairs-per-root", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-source-sha256", action="store_true")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def root_of(record: dict[str, Any]) -> str:
    return record["pair_id"].split("::", 1)[0]


def group_pairs(records: list[dict[str, Any]]) -> dict[str, list[list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["pair_id"]].append(record)
    result: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for pair_id, rows in grouped.items():
        roles = {row["role"] for row in rows}
        if len(rows) != 2 or roles != {"bug", "clean_or_fixed"}:
            raise ValueError(f"incomplete pair: {pair_id}")
        if len({row["scene_id"] for row in rows}) != 1:
            raise ValueError(f"pair scene mismatch: {pair_id}")
        result[root_of(rows[0])].append(sorted(rows, key=lambda row: row["role"]))
    return result


def select_pairs(
    records: list[dict[str, Any]], pairs_per_root: int | None, seed: int
) -> list[list[dict[str, Any]]]:
    by_root = group_pairs(records)
    selected = []
    for root in sorted(by_root):
        values = sorted(by_root[root], key=lambda pair: pair[0]["pair_id"])
        if pairs_per_root is not None:
            by_condition: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
            for pair in values:
                by_condition[pair[0]["camera_condition"]].append(pair)
            stationary_count = pairs_per_root // 2
            moving_count = pairs_per_root - stationary_count
            requested = {
                "stationary_camera": stationary_count,
                "moving_camera": moving_count,
            }
            balanced = []
            for condition, count in requested.items():
                condition_values = by_condition[condition]
                random.Random(f"{seed}:{root}:{condition}").shuffle(condition_values)
                if len(condition_values) < count:
                    raise ValueError(
                        f"{root}/{condition}: requested {count}, found {len(condition_values)}"
                    )
                balanced.extend(condition_values[:count])
            values = balanced
        else:
            random.Random(f"{seed}:{root}").shuffle(values)
        selected.extend(values)
    random.Random(seed).shuffle(selected)
    return selected


def window_starts(frame_count: int, window: int, stride: int) -> list[int]:
    if frame_count <= 0 or window <= 0 or stride <= 0:
        raise ValueError("frame_count, window and stride must be positive")
    if frame_count <= window:
        return [0]
    starts = list(range(0, frame_count - window + 1, stride))
    final = frame_count - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def frame_targets(record: dict[str, Any]) -> np.ndarray:
    target = np.zeros(int(record["frame_count"]), dtype=bool)
    if record["role"] == "bug":
        for event in record["answer"]["events"]:
            start = max(0, int(event["start_frame"]))
            end = min(len(target) - 1, int(event["end_frame"]))
            target[start : end + 1] = True
    return target


def window_targets(record: dict[str, Any], window: int, stride: int) -> tuple[list[int], np.ndarray]:
    frames = frame_targets(record)
    starts = window_starts(len(frames), window, stride)
    labels = np.asarray(
        [bool(frames[start : min(start + window, len(frames))].any()) for start in starts],
        dtype=np.float32,
    )
    return starts, labels


def oracle_pair_roi(pair: list[dict[str, Any]], margin_ratio: float, min_side: int) -> list[int]:
    bug = next(row for row in pair if row["role"] == "bug")
    width, height = int(bug["width"]), int(bug["height"])
    manifest = json.loads(Path(bug["effect_mask_manifest"]).read_text(encoding="utf-8"))
    boxes = [item.get("bbox_xyxy") for item in manifest.get("effect_masks", [])]
    boxes = [box for box in boxes if box and len(box) == 4]
    if not boxes:
        return [0, 0, width, height]
    x1 = min(int(box[0]) for box in boxes)
    y1 = min(int(box[1]) for box in boxes)
    x2 = max(int(box[2]) for box in boxes)
    y2 = max(int(box[3]) for box in boxes)
    crop_width = max(min_side, math.ceil((x2 - x1) * (1 + 2 * margin_ratio)))
    crop_height = max(min_side, math.ceil((y2 - y1) * (1 + 2 * margin_ratio)))
    crop_width, crop_height = min(crop_width, width), min(crop_height, height)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left = max(0, min(width - crop_width, round(center_x - crop_width / 2)))
    top = max(0, min(height - crop_height, round(center_y - crop_height / 2)))
    return [left, top, left + crop_width, top + crop_height]


def pair_roi(pair: list[dict[str, Any]], options: argparse.Namespace) -> list[int] | None:
    if options.branch == "global":
        return None
    return oracle_pair_roi(pair, options.roi_margin_ratio, options.roi_min_side)


def apply_roi(video: np.ndarray, roi: list[int] | None) -> np.ndarray:
    if roi is None:
        return video
    x1, y1, x2, y2 = roi
    cropped = video[:, y1:y2, x1:x2]
    if not cropped.size:
        raise ValueError(f"empty Oracle ROI: {roi}")
    return np.ascontiguousarray(cropped)


def decode_video(record: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    import av

    path = Path(record["video_path"])
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else float(record["fps"])
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"empty video: {path}")
    video = np.stack(frames)
    observed = tuple(int(value) for value in video.shape[:3])
    expected = (int(record["frame_count"]), int(record["height"]), int(record["width"]))
    if observed != expected:
        raise RuntimeError(f"{record['sample_id']}: decoded {observed}, expected {expected}")
    if not math.isclose(fps, float(record["fps"]), rel_tol=0.0, abs_tol=1e-4):
        raise RuntimeError(f"{record['sample_id']}: decoded FPS {fps}, expected {record['fps']}")
    return video, {
        "video_path": str(path),
        "decoded_frame_count": observed[0],
        "decoded_height": observed[1],
        "decoded_width": observed[2],
        "decoded_fps": fps,
        "all_frames": True,
        "native_resolution_input": True,
        "spatial_resize": False,
        "temporal_sampling": False,
        "video_reencode": False,
    }


def normalized_window(video: np.ndarray, start: int, window_frames: int, device):
    import torch

    clip = video[start : start + window_frames]
    valid = len(clip)
    if valid < window_frames:
        padding = np.repeat(clip[-1:], window_frames - valid, axis=0)
        clip = np.concatenate((clip, padding), axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(clip)).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0).div_(255.0)
    mean = torch.as_tensor(KINETICS_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.as_tensor(KINETICS_STD, device=device).view(1, 3, 1, 1, 1)
    return tensor.sub_(mean).div_(std), valid


def create_model(options: argparse.Namespace):
    import torch
    from torchvision.models.video import R3D_18_Weights, r3d_18

    weights = None if options.no_pretrained else R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights, progress=True)
    model.stem[0].stride = (1, options.stem_spatial_stride, options.stem_spatial_stride)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    torch.nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    torch.nn.init.zeros_(model.fc.bias)
    return model.to("cuda:0")


def auc_score(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    predicted = [score >= threshold for score in scores]
    tp = sum(label == 1 and pred for label, pred in zip(labels, predicted))
    fp = sum(label == 0 and pred for label, pred in zip(labels, predicted))
    fn = sum(label == 1 and not pred for label, pred in zip(labels, predicted))
    tn = sum(label == 0 and not pred for label, pred in zip(labels, predicted))
    accuracy = (tp + tn) / len(labels) if labels else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": auc_score(labels, scores),
    }


def calibrate_threshold(labels: list[int], scores: list[float]) -> dict[str, Any]:
    candidates = sorted({0.0, 1.0, *scores, *((a + b) / 2 for a, b in zip(sorted(scores), sorted(scores)[1:]))})
    rows = [binary_metrics(labels, scores, threshold) for threshold in candidates]
    return max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"] if row["balanced_accuracy"] is not None else -1,
            row["f1"],
            row["specificity"] if row["specificity"] is not None else -1,
        ),
    )


def aggregate_top_k(probabilities: list[float], top_k: int) -> float:
    if not probabilities:
        raise ValueError("empty probability list")
    count = min(top_k, len(probabilities))
    return statistics.fmean(sorted(probabilities, reverse=True)[:count])


def infer_video(model, record, options, video: np.ndarray | None = None):
    import torch

    if video is None:
        video, audit = decode_video(record)
    else:
        audit = {"decoded_frame_count": len(video), "reused_pair_decode": True}
    starts, labels = window_targets(record, options.window_frames, options.window_stride)
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in starts:
            tensor, _ = normalized_window(video, start, options.window_frames, "cuda:0")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(tensor).squeeze()
            probabilities.append(float(torch.sigmoid(logit.float()).cpu()))
            del tensor, logit
    return {
        "sample_id": record["sample_id"],
        "pair_id": record["pair_id"],
        "role": record["role"],
        "root_cause": root_of(record),
        "camera_condition": record["camera_condition"],
        "fps": float(record["fps"]),
        "frame_count": int(record["frame_count"]),
        "window_starts": starts,
        "window_end_frames": [min(start + options.window_frames, len(video)) - 1 for start in starts],
        "window_probabilities": probabilities,
        "window_targets": labels.astype(int).tolist(),
        "frame_targets": frame_targets(record).astype(int).tolist(),
        "video_score": aggregate_top_k(probabilities, options.top_k),
        "expected_positive": record["role"] == "bug",
        "input_audit": audit,
    }


def train_video(model, optimizer, scaler, record, video, positive_weight, options):
    import torch
    import torch.nn.functional as functional

    starts, labels = window_targets(record, options.window_frames, options.window_stride)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    probabilities = []
    for start, label in zip(starts, labels):
        tensor, _ = normalized_window(video, start, options.window_frames, "cuda:0")
        target = torch.tensor([label], device="cuda:0", dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(tensor).view(1)
            weight = torch.where(target > 0, positive_weight, torch.ones_like(target))
            loss = functional.binary_cross_entropy_with_logits(logit.float(), target, weight=weight)
            scaled_loss = loss / len(starts)
        scaler.scale(scaled_loss).backward()
        losses.append(float(loss.detach().cpu()))
        probabilities.append(float(torch.sigmoid(logit.detach().float()).cpu()))
        del tensor, target, logit, loss, scaled_loss
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip)
    scaler.step(optimizer)
    scaler.update()
    return statistics.fmean(losses), aggregate_top_k(probabilities, options.top_k), len(starts)


def train_pair(model, optimizer, pair, videos, positive_weight, options):
    

    import torch
    import torch.nn.functional as functional

    bug = next(record for record in pair if record["role"] == "bug")
    clean = next(record for record in pair if record["role"] == "clean_or_fixed")
    bug_starts, bug_labels = window_targets(
        bug, options.window_frames, options.window_stride
    )
    clean_starts, clean_labels = window_targets(
        clean, options.window_frames, options.window_stride
    )
    if bug_starts != clean_starts or clean_labels.any():
        raise ValueError(f"invalid aligned pair targets: {bug['pair_id']}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    ranking_losses = []
    bug_probabilities, clean_probabilities = [], []
    for start, bug_label in zip(bug_starts, bug_labels):
        bug_tensor, _ = normalized_window(
            videos[bug["sample_id"]], start, options.window_frames, "cuda:0"
        )
        clean_tensor, _ = normalized_window(
            videos[clean["sample_id"]], start, options.window_frames, "cuda:0"
        )
        inputs = torch.cat((bug_tensor, clean_tensor), dim=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs).view(2).float()
            targets = torch.tensor([bug_label, 0.0], device="cuda:0")
            weights = torch.where(targets > 0, positive_weight, torch.ones_like(targets))
            classification = functional.binary_cross_entropy_with_logits(
                logits, targets, weight=weights
            )
            if bug_label > 0:
                ranking = functional.relu(
                    options.pair_ranking_margin_logit - (logits[0] - logits[1])
                )
            else:
                ranking = logits.new_zeros(())
            loss = classification + options.pair_ranking_weight * ranking
            (loss / len(bug_starts)).backward()
        probabilities = torch.sigmoid(logits.detach()).cpu().tolist()
        bug_probabilities.append(float(probabilities[0]))
        clean_probabilities.append(float(probabilities[1]))
        losses.append(float(classification.detach().cpu()))
        ranking_losses.append(float(ranking.detach().cpu()))
        del bug_tensor, clean_tensor, inputs, logits, targets, weights, loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip)
    optimizer.step()
    return {
        "classification_loss": statistics.fmean(losses),
        "ranking_loss": statistics.fmean(ranking_losses),
        "bug_score": aggregate_top_k(bug_probabilities, options.top_k),
        "clean_score": aggregate_top_k(clean_probabilities, options.top_k),
        "positive_bug_window_margin_fraction": (
            sum(
                bug_probability > clean_probability
                for bug_probability, clean_probability, label in zip(
                    bug_probabilities, clean_probabilities, bug_labels
                )
                if label > 0
            )
            / max(1, int(bug_labels.sum()))
        ),
        "windows": len(bug_starts),
    }


def temporal_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    native = all("frame_targets" in row for row in rows)
    if any("frame_targets" in row for row in rows) and not native:
        raise ValueError("cannot mix native and window-expanded temporal ground truth")
    positive_ious = []
    false_events = 0
    clean_minutes = 0.0
    for row in rows:
        predicted = np.zeros(row["frame_count"], dtype=bool)
        for start, end, score in zip(
            row["window_starts"], row["window_end_frames"], row["window_probabilities"]
        ):
            if score >= threshold:
                predicted[start : end + 1] = True
        expected = np.zeros(row["frame_count"], dtype=bool)
        if native:
            expected = np.asarray(row["frame_targets"], dtype=bool)
            if expected.shape != predicted.shape:
                raise ValueError("native temporal labels must match frame_count")
        else:
            
            for start, end, value in zip(
                row["window_starts"], row["window_end_frames"], row["window_targets"]
            ):
                if value:
                    expected[start : end + 1] = True
        if row["expected_positive"]:
            union = np.logical_or(predicted, expected).sum()
            positive_ious.append(float(np.logical_and(predicted, expected).sum() / union) if union else 1.0)
        else:
            clean_minutes += row["frame_count"] / row["fps"] / 60
            active = np.flatnonzero(predicted).tolist()
            false_events += sum(index == 0 or not predicted[index - 1] for index in active)
    return {
        ("mean_native_gt_temporal_iou_positive" if native else
         "mean_coarse_temporal_iou_positive"): statistics.fmean(positive_ious) if positive_ious else None,
        "ground_truth_resolution": "native_source_frame" if native else "legacy_window_expanded",
        "false_positive_events_per_clean_minute": false_events / clean_minutes if clean_minutes else None,
        "temporal_resolution": "overlapping_32_frame_windows",
    }


def main() -> None:
    options = parse_args()
    if options.epochs <= 0 or options.window_frames <= 0 or options.window_stride <= 0:
        raise SystemExit("epochs and window dimensions must be positive")
    if options.window_stride > options.window_frames:
        raise SystemExit("window stride cannot exceed window length")
    if options.stem_spatial_stride not in (2, 4, 8):
        raise SystemExit("stem spatial stride must be 2, 4 or 8")
    import torch

    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    train_records = read_jsonl(options.train_manifest)
    validation_records = read_jsonl(options.validation_manifest)
    train_pairs = select_pairs(train_records, options.train_pairs_per_root, options.seed)
    validation_pairs = select_pairs(
        validation_records, options.validation_pairs_per_root, options.seed + 1
    )
    train_flat = [record for pair in train_pairs for record in pair]
    validation_flat = [record for pair in validation_pairs for record in pair]
    train_scenes = {row["scene_id"] for row in train_flat}
    validation_scenes = {row["scene_id"] for row in validation_flat}
    if train_scenes & validation_scenes:
        raise RuntimeError("scene leakage between train and validation")
    positive_windows = negative_windows = 0
    for record in train_flat:
        _, labels = window_targets(record, options.window_frames, options.window_stride)
        positive_windows += int(labels.sum())
        negative_windows += int(len(labels) - labels.sum())
    positive_weight_value = min(20.0, negative_windows / max(1, positive_windows))
    positive_weight = torch.tensor(positive_weight_value, device="cuda:0")

    config = {
        "schema_version": "1.0.0",
        "protocol": PROTOCOL,
        "created_utc": utc_now(),
        "train_manifest": str(options.train_manifest.resolve()),
        "train_manifest_sha256": sha256_file(options.train_manifest),
        "validation_manifest": str(options.validation_manifest.resolve()),
        "validation_manifest_sha256": sha256_file(options.validation_manifest),
        "test_used": False,
        "epochs": options.epochs,
        "window_frames": options.window_frames,
        "window_stride": options.window_stride,
        "video_aggregation": f"mean_top_{options.top_k}_window_probabilities",
        "model_branch": options.branch,
        "native_input": True,
        "spatial_resize": False,
        "temporal_sampling": False,
        "stem_spatial_stride": options.stem_spatial_stride,
        "oracle_roi": (
            {
                "source": "union of native per-frame causal-effect mask bounding boxes",
                "same_fixed_crop_for_bug_and_clean": True,
                "margin_ratio": options.roi_margin_ratio,
                "minimum_side_pixels": options.roi_min_side,
                "deployment_status": "oracle_ablation_not_standalone_detector",
            }
            if options.branch == "oracle_roi"
            else None
        ),
        "pretrained_kinetics400": not options.no_pretrained,
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "train_video_count": len(train_flat),
        "validation_video_count": len(validation_flat),
        "train_pair_camera_counts": dict(
            sorted(
                (condition, sum(pair[0]["camera_condition"] == condition for pair in train_pairs))
                for condition in ("stationary_camera", "moving_camera")
            )
        ),
        "validation_pair_camera_counts": dict(
            sorted(
                (
                    condition,
                    sum(pair[0]["camera_condition"] == condition for pair in validation_pairs),
                )
                for condition in ("stationary_camera", "moving_camera")
            )
        ),
        "positive_window_count": positive_windows,
        "negative_window_count": negative_windows,
        "positive_window_loss_weight": positive_weight_value,
        "pair_ranking_weight": options.pair_ranking_weight,
        "pair_ranking_margin_logit": options.pair_ranking_margin_logit,
        "pairwise_causal_training": options.pair_ranking_weight > 0,
        "learning_rate": options.learning_rate,
        "weight_decay": options.weight_decay,
        "seed": options.seed,
        "source_sha256_checked": not options.skip_source_sha256,
    }
    (options.output_dir / "training_protocol.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model = create_model(options)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history = []
    for epoch in range(1, options.epochs + 1):
        epoch_losses = []
        epoch_ranking_losses = []
        seen_windows = 0
        for pair_index, pair in enumerate(train_pairs, start=1):
            roi = pair_roi(pair, options)
            videos = {}
            for record in pair:
                if not options.skip_source_sha256 and record.get("video_sha256"):
                    if sha256_file(Path(record["video_path"])) != record["video_sha256"]:
                        raise RuntimeError(f"source hash mismatch: {record['sample_id']}")
                video, _ = decode_video(record)
                video = apply_roi(video, roi)
                videos[record["sample_id"]] = video
            outcome = train_pair(
                model, optimizer, pair, videos, positive_weight, options
            )
            epoch_losses.append(outcome["classification_loss"])
            epoch_ranking_losses.append(outcome["ranking_loss"])
            seen_windows += outcome["windows"] * 2
            print(
                canonical_json(
                    {
                        "epoch": epoch,
                        "pair_progress": f"{pair_index}/{len(train_pairs)}",
                        "pair_id": pair[0]["pair_id"],
                        "classification_loss": round(
                            outcome["classification_loss"], 6
                        ),
                        "ranking_loss": round(outcome["ranking_loss"], 6),
                        "bug_score": round(outcome["bug_score"], 6),
                        "clean_score": round(outcome["clean_score"], 6),
                        "positive_bug_window_margin_fraction": round(
                            outcome["positive_bug_window_margin_fraction"], 6
                        ),
                        "windows_per_video": outcome["windows"],
                        "oracle_roi_xyxy": roi,
                    }
                ),
                flush=True,
            )
            del videos
        predictions = []
        for pair in validation_pairs:
            roi = pair_roi(pair, options)
            for record in pair:
                video, audit = decode_video(record)
                video = apply_roi(video, roi)
                prediction = infer_video(model, record, options, video=video)
                prediction["input_audit"] = audit
                prediction["oracle_roi_xyxy"] = roi
                predictions.append(prediction)
                del video
        labels = [int(row["expected_positive"]) for row in predictions]
        scores = [float(row["video_score"]) for row in predictions]
        calibration = calibrate_threshold(labels, scores)
        calibration["temporal"] = temporal_metrics(predictions, calibration["threshold"])
        moving_clean = [
            row for row in predictions
            if not row["expected_positive"] and row["camera_condition"] == "moving_camera"
        ]
        calibration["moving_clean_specificity"] = (
            sum(row["video_score"] < calibration["threshold"] for row in moving_clean)
            / len(moving_clean)
            if moving_clean
            else None
        )
        epoch_result = {
            "epoch": epoch,
            "mean_training_window_loss": statistics.fmean(epoch_losses),
            "mean_training_pair_ranking_loss": statistics.fmean(
                epoch_ranking_losses
            ),
            "training_window_count": seen_windows,
            "validation": calibration,
        }
        history.append(epoch_result)
        checkpoint = options.output_dir / f"checkpoint-{epoch:04d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint / "model.pt")
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        (checkpoint / "validation_predictions.jsonl").write_text(
            "".join(canonical_json(row) + "\n" for row in predictions), encoding="utf-8"
        )
        (checkpoint / "metrics.json").write_text(
            json.dumps(epoch_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(canonical_json({"epoch_complete": epoch, "validation": calibration}), flush=True)
    result = {
        "schema_version": "1.0.0",
        "protocol": PROTOCOL,
        "completed_utc": utc_now(),
        "status": "completed",
        "history": history,
        "test_used": False,
    }
    (options.output_dir / "run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
