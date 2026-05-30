#!/usr/bin/env python3
"""Grad-CAM aggregation for a synthetic-only checkpoint on real and synthetic splits."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_allnew_no_rpm_diagnostics import (  # noqa: E402
    import_attr,
    load_state,
    make_model,
    names_to_list,
    resolve_checkpoint,
)
from scripts.run_no_rpm_window30_gradcam import Conv3DGradCam, normalize, overlay_frame  # noqa: E402


THRESHOLDS = (0.25, 0.50, 0.75)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return (index, *self.dataset[index])


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as file:
        return yaml.safe_load(file)


def load_manifest_pairs(manifest_path: str) -> tuple[list[str], list[str]]:
    with Path(manifest_path).open("r") as file:
        payload = json.load(file)
    records = payload["samples"] if isinstance(payload, dict) else payload
    return [record["video_path"] for record in records], [record["parameters_norm_path"] for record in records]


def split_dataset_paths(config: dict[str, Any], split: str) -> tuple[list[str], list[str], dict[str, Any], str, Path]:
    video_subdir = config["misc_dir"]["video_subdir"]
    norm_subdir = config["misc_dir"]["norm_subdir"]

    if split == "real_test":
        section = config["dataset"]["test"]
        manifest = section.get("manifest")
        if manifest:
            video_paths, para_paths = load_manifest_pairs(manifest)
            root = Path(section["test_root"])
        else:
            root = Path(section["test_root"])
            video_paths = sorted(str(path) for path in (root / video_subdir).glob("*.mp4"))
            para_paths = sorted(str(path) for path in (root / norm_subdir).glob("*.json"))
        return video_paths, para_paths, section, section["dataloader"]["dataloader"], root

    if split == "synthetic_val":
        section = config["dataset"]["train"]
        manifest = section.get("manifest")
        if manifest:
            video_paths, para_paths = load_manifest_pairs(manifest)
            root = Path(section["train_root"])
        else:
            root = Path(section["train_root"])
            video_paths = sorted(str(path) for path in (root / video_subdir).glob("*.mp4"))
            para_paths = sorted(str(path) for path in (root / norm_subdir).glob("*.json"))

        if bool(section.get("use_all_samples", False)):
            return video_paths[:1], para_paths[:1], section, section["dataloader"]["dataloader"], root

        test_size = float(section["dataloader"]["test_size"])
        random_state = int(section["dataloader"]["random_state"])
        _, val_video_paths = train_test_split(video_paths, test_size=test_size, random_state=random_state)
        _, val_para_paths = train_test_split(para_paths, test_size=test_size, random_state=random_state)
        return val_video_paths, val_para_paths, section, section["dataloader"]["dataloader"], root

    raise ValueError(f"Unsupported split: {split}")


def build_loader(
    config: dict[str, Any],
    split: str,
    batch_size: int,
    max_samples: int | None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> tuple[DataLoader, list[str], Path, int, int]:
    video_paths, para_paths, section, dataset_name, root = split_dataset_paths(config, split)
    dataset_class = import_attr(f"datasets.{dataset_name}", dataset_name)
    dataset = dataset_class(
        video_paths,
        para_paths,
        float(section["frame_num"]),
        float(section["time"]),
        aug_bool=False,
        visc_class=int(config["model"]["transformer"]["class"]),
    )
    indices = list(range(len(dataset)))
    if max_samples is not None:
        indices = indices[: int(max_samples)]
    total_before_shard = len(indices)
    if int(num_shards) > 1:
        indices = indices[int(shard_index) :: int(num_shards)]
    loader = DataLoader(
        Subset(IndexedDataset(dataset), indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )
    return loader, para_paths, root, len(indices), total_before_shard


def load_param_metadata(path: str, split: str, class_count: int) -> dict[str, Any]:
    with Path(path).open("r") as file:
        payload = json.load(file)
    visc_index = int(payload["visc_index"])
    if split == "real_test":
        class_width = max(1, 10 // class_count)
    else:
        class_width = max(1, 50 // class_count)
    return {
        "pattern_id": str(payload.get("background", "unknown")),
        "visc_index": visc_index,
        "true_viscosity_bin": visc_index // class_width,
        "kinematic_viscosity_norm": float(payload["kinematic_viscosity"]),
        "rpm_idx": int(payload.get("rpm_idx", -1)),
    }


def load_background(root: Path, pattern_id: str, image_size: int) -> np.ndarray:
    path = root / "backgrounds" / f"{pattern_id}.png"
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    top = max(0, (height - image_size) // 2)
    left = max(0, (width - image_size) // 2)
    image = image[top : top + image_size, left : left + image_size]
    if image.shape[:2] != (image_size, image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def entropy_norm(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(np.nansum(flat))
    if total <= 0:
        return 0.0
    probs = flat / total
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / math.log(probs.size))


def spatial_center_distance(values: np.ndarray) -> float:
    spatial = np.asarray(values, dtype=np.float64)
    total = float(np.nansum(spatial))
    if total <= 0:
        return 0.0
    probs = spatial / total
    yy, xx = np.mgrid[0 : spatial.shape[0], 0 : spatial.shape[1]]
    cy = float(np.nansum(yy * probs))
    cx = float(np.nansum(xx * probs))
    center_y = (spatial.shape[0] - 1) / 2.0
    center_x = (spatial.shape[1] - 1) / 2.0
    max_dist = math.hypot(center_y, center_x)
    return math.hypot(cy - center_y, cx - center_x) / max_dist if max_dist > 0 else 0.0


def threshold_suffix(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def active_area_metrics(mean_map: np.ndarray, sample_count: int) -> dict[str, Any]:
    if sample_count <= 0:
        row: dict[str, Any] = {
            "sample_count": 0,
            "grid_height": None,
            "grid_width": None,
            "mean_cam": None,
            "max_cam": None,
            "spatial_entropy": None,
            "center_distance": None,
        }
        for threshold in THRESHOLDS:
            suffix = threshold_suffix(threshold)
            row[f"active_tokens_t{suffix}"] = None
            row[f"active_area_fraction_t{suffix}"] = None
            row[f"cam_mass_share_t{suffix}"] = None
        return row

    norm_map = normalize(mean_map)
    total_tokens = int(norm_map.size)
    row = {
        "sample_count": int(sample_count),
        "grid_height": int(norm_map.shape[0]),
        "grid_width": int(norm_map.shape[1]),
        "mean_cam": float(np.mean(norm_map)),
        "max_cam": float(np.max(norm_map)),
        "spatial_entropy": entropy_norm(norm_map),
        "center_distance": spatial_center_distance(norm_map),
    }
    for threshold in THRESHOLDS:
        mask = norm_map >= threshold
        active_tokens = int(mask.sum())
        area_fraction = active_tokens / total_tokens if total_tokens else 0.0
        mass_share = float(norm_map[mask].sum() / (norm_map.sum() + 1e-8)) if active_tokens else 0.0
        suffix = threshold_suffix(threshold)
        row[f"active_tokens_t{suffix}"] = active_tokens
        row[f"active_area_fraction_t{suffix}"] = float(area_fraction)
        row[f"cam_mass_share_t{suffix}"] = mass_share
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["empty"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rpm_for_model(rpm_idx: torch.Tensor, rpm_bool: bool) -> torch.Tensor:
    rpm_idx = rpm_idx.long().view(-1)
    return rpm_idx if rpm_bool else torch.zeros_like(rpm_idx)


def confidence_for_predictions(logits: torch.Tensor, predictions: torch.Tensor) -> list[float]:
    probs = F.softmax(logits.detach().cpu().float(), dim=1)
    return [float(probs[row, int(pred)].item()) for row, pred in enumerate(predictions.detach().cpu().view(-1))]


def plot_pattern_bin_pages(
    *,
    output_dir: Path,
    split: str,
    pattern_ids: list[str],
    class_count: int,
    aggregate_maps: dict[tuple[str, int], np.ndarray],
    area_rows_by_key: dict[tuple[str, int], dict[str, Any]],
    backgrounds: dict[str, np.ndarray],
    patterns_per_page: int,
) -> list[str]:
    page_paths: list[str] = []
    for page_index, start in enumerate(range(0, len(pattern_ids), patterns_per_page), start=1):
        page_patterns = pattern_ids[start : start + patterns_per_page]
        fig, axes = plt.subplots(
            len(page_patterns),
            class_count,
            figsize=(2.05 * class_count, 2.05 * len(page_patterns)),
            squeeze=False,
        )
        for row_index, pattern_id in enumerate(page_patterns):
            background = backgrounds[pattern_id]
            for viscosity_bin in range(class_count):
                ax = axes[row_index, viscosity_bin]
                key = (pattern_id, viscosity_bin)
                row = area_rows_by_key[key]
                mean_map = aggregate_maps.get(key)
                if mean_map is None or int(row["sample_count"]) == 0:
                    ax.axis("off")
                    ax.set_title(f"B{viscosity_bin}\nn=0", fontsize=7)
                    continue
                norm_map = normalize(mean_map)
                tile = overlay_frame(background, norm_map, alpha=0.48)
                ax.imshow(tile)
                mask = cv2.resize(
                    (norm_map >= 0.75).astype(np.float32),
                    (background.shape[1], background.shape[0]),
                    interpolation=0,
                )
                if np.any(mask):
                    ax.contour(mask, levels=[0.5], colors="white", linewidths=0.6)
                    ax.contour(mask, levels=[0.5], colors="black", linewidths=0.25)
                accuracy = row.get("accuracy")
                accuracy_text = "" if accuracy is None else f"\nacc={100.0 * float(accuracy):.1f}%"
                ax.set_title(
                    f"B{viscosity_bin} n={int(row['sample_count'])}{accuracy_text}",
                    fontsize=6.7,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if row_index == 0:
                    ax.set_xlabel(f"bin {viscosity_bin}", fontsize=8)
            axes[row_index, 0].set_ylabel(f"Pattern {pattern_id}", fontsize=9)
        fig.suptitle(
            f"{split}: synthetic-only checkpoint Grad-CAM by pattern and true viscosity bin",
            fontsize=13,
        )
        fig.text(
            0.5,
            0.978,
            "Rows are background patterns; columns are 10 true viscosity bins. White contour is CAM >= 0.75.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        output_path = output_dir / f"{split}_pattern_bin_gradcam_grid_page_{page_index:02d}.png"
        fig.savefig(output_path, dpi=190)
        plt.close(fig)
        page_paths.append(str(output_path))
    return page_paths


def plot_area_heatmap(
    output_path: Path,
    split: str,
    pattern_ids: list[str],
    class_count: int,
    area_rows_by_key: dict[tuple[str, int], dict[str, Any]],
    field: str,
    title: str,
) -> None:
    matrix = np.full((len(pattern_ids), class_count), np.nan, dtype=np.float32)
    for row_idx, pattern_id in enumerate(pattern_ids):
        for viscosity_bin in range(class_count):
            value = area_rows_by_key[(pattern_id, viscosity_bin)].get(field)
            if value is not None and value != "":
                matrix[row_idx, viscosity_bin] = float(value)
    if field.startswith("active_area_fraction"):
        matrix *= 100.0
    fig_height = max(4.5, 0.24 * len(pattern_ids))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(class_count))
    ax.set_xticklabels([str(idx) for idx in range(class_count)])
    ax.set_yticks(np.arange(len(pattern_ids)))
    ax.set_yticklabels([str(pattern) for pattern in pattern_ids], fontsize=6 if len(pattern_ids) > 20 else 8)
    ax.set_xlabel("true viscosity bin")
    ax.set_ylabel("pattern")
    ax.set_title(f"{split}: {title}")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("% active area" if field.startswith("active_area_fraction") else field)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rebuild/retries/allnew_synth_no_rpm_augv1_realtest_frozen_eval.yaml")
    parser.add_argument("--checkpoint", default="allnew_synthetic_pretrain_sph35000_no_rpm_augv1_ep80.pth")
    parser.add_argument("--split", choices=["real_test", "synthetic_val"], required=True)
    parser.add_argument(
        "--output-root",
        default="outputs/rebuild_reproduction/synthetic_weight_gradcam_bins/allnew_synth_no_rpm_augv1",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--patterns-per-page", type=int, default=10)
    parser.add_argument("--target-layer", default="featureextractor.embeddings.patch_embeddings.projection")
    parser.add_argument("--gradcam-target", choices=["prediction", "true"], default="prediction")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    config = load_config(Path(args.config))
    if bool(config["model"]["embeddings"]["rpm_bool"]):
        raise ValueError("This analysis expects a synthetic-only no-RPM checkpoint config.")

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    class_count = int(config["model"]["transformer"]["class"])
    image_size = int(config["model"]["transformer"].get("image_size", 224))
    base_output_dir = Path(args.output_root) / args.split
    if int(args.num_shards) > 1:
        output_dir = base_output_dir / "shards" / f"shard_{int(args.shard_index):02d}_of_{int(args.num_shards):02d}"
    else:
        output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir = output_dir / "aggregate_maps"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    loader, para_paths, dataset_root, shard_selected, total_selected = build_loader(
        config,
        args.split,
        args.batch_size,
        args.max_samples,
        int(args.num_shards),
        int(args.shard_index),
    )
    checkpoint = resolve_checkpoint(config, args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = make_model(config, device, output_attentions=False)
    load_info = load_state(model, checkpoint, device)
    model.eval()
    cam = Conv3DGradCam(model, args.target_layer)

    rpm_bool = bool(config["model"]["embeddings"]["rpm_bool"])
    group_sums: dict[tuple[str, int], np.ndarray] = {}
    group_counts: dict[tuple[str, int], int] = {}
    group_correct: dict[tuple[str, int], int] = {}
    sample_rows: list[dict[str, Any]] = []
    all_pattern_ids: set[str] = set()
    rpm_model_nonzero_count = 0

    try:
        for sample_indices, frames, _parameters, hotvector, names, rpm_idx, pattern in loader:
            frames = frames.to(device)
            labels = hotvector.to(device).long().view(-1)
            rpm_actual = rpm_idx.long().view(-1)
            rpm_model = rpm_for_model(rpm_idx.to(device), rpm_bool)
            pattern = pattern.to(device)
            rpm_model_nonzero_count += int((rpm_model.detach().cpu() != 0).sum().item())

            model.zero_grad(set_to_none=True)
            logits = model(frames, rpm_model, pattern)
            predictions = logits.argmax(dim=1)
            targets = predictions if args.gradcam_target == "prediction" else labels
            score = logits[torch.arange(logits.shape[0], device=device), targets].sum()
            score.backward()
            volumes = cam.volumes()
            confidences = confidence_for_predictions(logits, predictions)
            name_list = names_to_list(names)

            for batch_idx, sample_index in enumerate(sample_indices.tolist()):
                metadata = load_param_metadata(para_paths[int(sample_index)], args.split, class_count)
                pattern_id = str(metadata["pattern_id"])
                true_bin = int(labels[batch_idx].detach().cpu().item())
                pred_bin = int(predictions[batch_idx].detach().cpu().item())
                target_bin = int(targets[batch_idx].detach().cpu().item())
                spatial = np.nanmean(volumes[batch_idx], axis=2).astype(np.float32)
                key = (pattern_id, true_bin)
                group_sums[key] = group_sums.get(key, np.zeros_like(spatial)) + spatial
                group_counts[key] = group_counts.get(key, 0) + 1
                group_correct[key] = group_correct.get(key, 0) + int(pred_bin == true_bin)
                all_pattern_ids.add(pattern_id)
                sample_rows.append(
                    {
                        "idx": int(sample_index),
                        "name": name_list[batch_idx],
                        "split": args.split,
                        "pattern_id": pattern_id,
                        "true_viscosity_bin": true_bin,
                        "metadata_true_viscosity_bin": int(metadata["true_viscosity_bin"]),
                        "prediction": pred_bin,
                        "correct": pred_bin == true_bin,
                        "target_bin": target_bin,
                        "confidence": confidences[batch_idx],
                        "visc_index": int(metadata["visc_index"]),
                        "kinematic_viscosity_norm": float(metadata["kinematic_viscosity_norm"]),
                        "rpm_idx": int(metadata["rpm_idx"]),
                        "spatial_cam_mean": float(np.mean(spatial)),
                        "spatial_cam_max": float(np.max(spatial)),
                    }
                )
    finally:
        cam.close()

    pattern_ids = sorted(all_pattern_ids, key=lambda value: int(value) if str(value).isdigit() else str(value))
    backgrounds = {pattern_id: load_background(dataset_root, pattern_id, image_size) for pattern_id in pattern_ids}
    aggregate_maps: dict[tuple[str, int], np.ndarray] = {}
    area_rows: list[dict[str, Any]] = []
    for pattern_id in pattern_ids:
        for viscosity_bin in range(class_count):
            key = (pattern_id, viscosity_bin)
            count = group_counts.get(key, 0)
            if count > 0:
                mean_map = group_sums[key] / count
                aggregate_maps[key] = mean_map
                np.save(
                    aggregate_dir / f"{args.split}_pattern_{safe_name(pattern_id)}_bin_{viscosity_bin}_mean_spatial_gradcam.npy",
                    mean_map.astype(np.float32),
                )
            metrics = active_area_metrics(aggregate_maps[key], count) if count > 0 else active_area_metrics(np.empty((0, 0)), 0)
            accuracy = (group_correct.get(key, 0) / count) if count > 0 else None
            area_rows.append(
                {
                    "split": args.split,
                    "pattern_id": pattern_id,
                    "true_viscosity_bin": viscosity_bin,
                    "accuracy": accuracy,
                    "correct_count": group_correct.get(key, 0) if count > 0 else 0,
                    **metrics,
                }
            )

    area_rows_by_key = {
        (str(row["pattern_id"]), int(row["true_viscosity_bin"])): row for row in area_rows
    }
    grid_pages = plot_pattern_bin_pages(
        output_dir=output_dir,
        split=args.split,
        pattern_ids=pattern_ids,
        class_count=class_count,
        aggregate_maps=aggregate_maps,
        area_rows_by_key=area_rows_by_key,
        backgrounds=backgrounds,
        patterns_per_page=max(1, int(args.patterns_per_page)),
    )
    area_heatmap = output_dir / f"{args.split}_active_area_t0p75_heatmap.png"
    plot_area_heatmap(
        area_heatmap,
        args.split,
        pattern_ids,
        class_count,
        area_rows_by_key,
        "active_area_fraction_t0p75",
        "CAM active area at threshold 0.75",
    )
    count_heatmap = output_dir / f"{args.split}_sample_count_heatmap.png"
    plot_area_heatmap(
        count_heatmap,
        args.split,
        pattern_ids,
        class_count,
        area_rows_by_key,
        "sample_count",
        "sample count by pattern and bin",
    )

    write_csv(output_dir / f"{args.split}_gradcam_sample_metrics.csv", sample_rows)
    write_csv(output_dir / f"{args.split}_pattern_bin_gradcam_area_summary.csv", area_rows)
    (output_dir / f"{args.split}_gradcam_sample_metrics.json").write_text(
        json.dumps(sample_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{args.split}_pattern_bin_gradcam_area_summary.json").write_text(
        json.dumps(area_rows, indent=2) + "\n",
        encoding="utf-8",
    )

    accuracy = float(np.mean([row["correct"] for row in sample_rows])) if sample_rows else None
    summary = {
        "split": args.split,
        "config": args.config,
        "checkpoint": str(checkpoint),
        "load_info": load_info,
        "device": str(device),
        "sample_count": len(sample_rows),
        "shard_samples_selected": shard_selected,
        "total_samples_selected": total_selected,
        "accuracy": accuracy,
        "pattern_count": len(pattern_ids),
        "pattern_ids": pattern_ids,
        "class_count": class_count,
        "thresholds": list(THRESHOLDS),
        "primary_threshold": 0.75,
        "threshold_basis": "per-pattern/bin accumulated mean Grad-CAM min-max normalized to [0,1]",
        "target_layer": args.target_layer,
        "gradcam_target": args.gradcam_target,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "base_output_dir": str(base_output_dir),
        "rpm_bool": rpm_bool,
        "rpm_input_policy": "actual_rpm_idx" if rpm_bool else "zero_tensor_for_model",
        "rpm_model_nonzero_count": rpm_model_nonzero_count,
        "grid_pages": grid_pages,
        "active_area_heatmap": str(area_heatmap),
        "sample_count_heatmap": str(count_heatmap),
        "area_summary_csv": str(output_dir / f"{args.split}_pattern_bin_gradcam_area_summary.csv"),
        "sample_metrics_csv": str(output_dir / f"{args.split}_gradcam_sample_metrics.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = [
        f"# Synthetic-Only Grad-CAM: {args.split}",
        "",
        f"- Config: `{args.config}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Split: `{args.split}`",
        f"- Samples: {len(sample_rows)}",
        f"- Accuracy: {accuracy if accuracy is not None else 'n/a'}",
        f"- Patterns: {len(pattern_ids)}",
        f"- Viscosity-bin axis: true bins `0..{class_count - 1}`",
        f"- Grad-CAM target: `{args.gradcam_target}` class score",
        f"- Target layer: `{args.target_layer}`",
        f"- RPM model input policy: `{summary['rpm_input_policy']}`",
        "",
        "## Outputs",
        "",
        f"- Summary: `{output_dir / 'summary.json'}`",
        f"- Area summary CSV: `{summary['area_summary_csv']}`",
        f"- Sample metrics CSV: `{summary['sample_metrics_csv']}`",
        f"- Active-area heatmap: `{area_heatmap}`",
        f"- Sample-count heatmap: `{count_heatmap}`",
    ]
    for page in grid_pages:
        report.append(f"- Pattern/bin grid: `{page}`")
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
