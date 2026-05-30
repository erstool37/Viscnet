#!/usr/bin/env python3
"""CPU-only Grad-CAM grouping by Re/We/Ca proxy bins for real-only weights."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
from collections import Counter
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_allnew_no_rpm_diagnostics import load_state, make_model, resolve_checkpoint  # noqa: E402
from scripts.run_no_rpm_window30_gradcam import (  # noqa: E402
    DEFAULT_RUN_DIR,
    Conv3DGradCam,
    WindowLoader,
    choose_window_record,
    load_config,
    load_json,
    normalize,
    overlay_frame,
    safe_name,
    split_dataset_paths,
)
from scripts.run_no_rpm_window30_gradcam_by_pattern import (  # noqa: E402
    active_area_metrics,
    load_background_image,
    load_pattern_id,
    threshold_area,
    write_csv,
)


PHYSICS_FEATURES = ("Re_proxy", "We_proxy", "Ca_proxy")
EXPECTED_FULL_SAMPLE_COUNT = 1000
EXPECTED_FULL_PATTERN_IDS = ("1", "2", "3", "4")


def load_raw_physics(test_root: Path, name: str) -> dict[str, float]:
    path = test_root / "parameters" / f"{name}.json"
    with path.open("r") as file:
        data = json.load(file)
    rho = float(data["density"])
    nu = float(data["kinematic_viscosity"])
    sigma = float(data["surface_tension"])
    rpm = float(data.get("RPM", data.get("rpm")))
    omega = 2.0 * math.pi * rpm / 60.0
    mu = rho * nu
    return {
        "density_raw": rho,
        "kinematic_viscosity_raw": nu,
        "surface_tension_raw": sigma,
        "rpm_raw": rpm,
        "omega_rad_s": omega,
        "Re_proxy": omega / nu,
        "We_proxy": rho * omega * omega / sigma,
        "Ca_proxy": mu * omega / sigma,
        "log10_Re_proxy": math.log10(omega / nu),
        "log10_We_proxy": math.log10(rho * omega * omega / sigma),
        "log10_Ca_proxy": math.log10(mu * omega / sigma),
    }


def assign_rank_bins(records: list[dict[str, Any]], feature: str, bin_count: int) -> list[dict[str, float | int]]:
    indexed = sorted(enumerate(records), key=lambda item: (float(item[1][feature]), int(item[0])))
    bins = [0 for _ in records]
    for rank, (index, _record) in enumerate(indexed):
        bins[index] = min(bin_count - 1, int(rank * bin_count / max(1, len(records))))
    ranges = []
    for bin_id in range(bin_count):
        values = [float(records[index][feature]) for index, value in enumerate(bins) if value == bin_id]
        ranges.append(
            {
                "feature": feature,
                "bin": bin_id,
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": float(np.mean(values)) if values else None,
            }
        )
    for record, bin_id in zip(records, bins):
        record[f"{feature}_bin"] = int(bin_id)
    return ranges


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_checkpoint_path(config: dict[str, Any], checkpoint_arg: str) -> Path:
    checkpoint = Path(checkpoint_arg)
    if checkpoint.exists() or checkpoint.is_absolute():
        return checkpoint
    return resolve_checkpoint(config, checkpoint_arg)


def validate_full_run_coverage(
    *,
    selected_count: int,
    pattern_ids: list[str],
    max_samples: int | None,
) -> dict[str, Any]:
    if max_samples is not None:
        return {
            "status": "skipped_max_samples",
            "reason": "--max-samples was set, so this is a bounded diagnostic run.",
        }

    failures = []
    if selected_count != EXPECTED_FULL_SAMPLE_COUNT:
        failures.append(f"selected_count={selected_count}, expected={EXPECTED_FULL_SAMPLE_COUNT}")
    if pattern_ids != list(EXPECTED_FULL_PATTERN_IDS):
        failures.append(f"pattern_ids={pattern_ids}, expected={list(EXPECTED_FULL_PATTERN_IDS)}")
    if failures:
        raise ValueError("Full-run coverage guard failed before Grad-CAM: " + "; ".join(failures))
    return {
        "status": "passed",
        "expected_sample_count": EXPECTED_FULL_SAMPLE_COUNT,
        "expected_pattern_ids": list(EXPECTED_FULL_PATTERN_IDS),
    }


def build_coverage_records(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for idx, record in enumerate(selected):
        row: dict[str, Any] = {
            "idx": idx,
            "name": str(record["name"]),
            "pattern_id": str(record["pattern_id"]),
            "true_viscosity_class": int(record["true_viscosity_class"]),
            "window_start": int(record["window_start"]),
        }
        for feature in PHYSICS_FEATURES:
            row[feature] = float(record[feature])
            row[f"log10_{feature}"] = float(record[f"log10_{feature}"])
            row[f"{feature}_bin"] = int(record[f"{feature}_bin"])
        records.append(row)
    return records


def write_coverage_audit(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    prediction_dir: Path,
    per_video_count: int,
    per_window_count: int,
    selected: list[dict[str, Any]],
    pattern_ids: list[str],
    bin_ranges_by_feature: dict[str, list[dict[str, Any]]],
    guard: dict[str, Any],
) -> dict[str, Any]:
    pattern_counts = Counter(str(record["pattern_id"]) for record in selected)
    class_counts = Counter(str(int(record["true_viscosity_class"])) for record in selected)
    records = build_coverage_records(selected)
    audit = {
        "status": "metadata_audited",
        "guard": guard,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "max_samples": args.max_samples,
        "source_per_video_count": per_video_count,
        "source_per_window_count": per_window_count,
        "selected_count": len(selected),
        "pattern_ids": pattern_ids,
        "pattern_counts": {
            pattern_id: int(pattern_counts.get(pattern_id, 0)) for pattern_id in pattern_ids
        },
        "true_viscosity_class_counts": {
            class_id: int(class_counts[class_id]) for class_id in sorted(class_counts, key=metric_sort_key)
        },
        "bin_count": int(args.bin_count),
        "bin_ranges": bin_ranges_by_feature,
        "selected_records": records,
    }
    json_path = output_dir / "coverage_audit.json"
    md_path = output_dir / "coverage_audit.md"
    write_json(json_path, audit)
    md_lines = [
        "# Grad-CAM Dimensionless Coverage Audit",
        "",
        f"- Status: `{guard['status']}`",
        f"- Config: `{args.config}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Prediction dir: `{prediction_dir}`",
        f"- Source per-video records: `{per_video_count}`",
        f"- Source per-window records: `{per_window_count}`",
        f"- Selected records: `{len(selected)}`",
        f"- Pattern ids: `{', '.join(pattern_ids)}`",
        f"- Max samples: `{args.max_samples}`",
        "",
        "## Pattern Counts",
        "",
        "| pattern_id | selected videos |",
        "| --- | ---: |",
    ]
    for pattern_id in pattern_ids:
        md_lines.append(f"| {pattern_id} | {int(pattern_counts.get(pattern_id, 0))} |")
    md_lines.extend(
        [
            "",
            "## Viscosity Class Counts",
            "",
            "| true_viscosity_class | selected videos |",
            "| --- | ---: |",
        ]
    )
    for class_id in sorted(class_counts, key=metric_sort_key):
        md_lines.append(f"| {class_id} | {int(class_counts[class_id])} |")
    md_lines.extend(
        [
            "",
            "## Per-Video Provenance",
            "",
            "The complete per-video list is in `coverage_audit.json` under `selected_records`.",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return audit


def metric_sort_key(value: str) -> int | str:
    return int(value) if str(value).isdigit() else str(value)


def threshold_mask_overlay(background: np.ndarray, mean_map: np.ndarray, threshold: float) -> np.ndarray:
    norm_map = normalize(mean_map)
    mask = cv2.resize(
        (norm_map >= float(threshold)).astype(np.float32),
        (background.shape[1], background.shape[0]),
        interpolation=0,
    )
    mask_rgb = np.zeros_like(background)
    mask_rgb[..., 0] = 1.0
    return np.clip(0.55 * background + 0.45 * mask_rgb * mask[..., None], 0.0, 1.0)


def plot_feature_grid(
    *,
    feature: str,
    output_path: Path,
    rows: list[dict[str, Any]],
    maps_by_key: dict[tuple[str, int], np.ndarray],
    backgrounds: dict[str, np.ndarray],
    bin_ranges: list[dict[str, Any]],
    bin_count: int,
    primary_threshold: float,
) -> None:
    pattern_ids = sorted(backgrounds.keys(), key=metric_sort_key)
    rows_by_key = {(str(row["pattern_id"]), int(row["bin"])): row for row in rows}
    fig, axes = plt.subplots(
        len(pattern_ids),
        bin_count,
        figsize=(2.05 * bin_count, 2.05 * len(pattern_ids)),
        squeeze=False,
    )
    for row_idx, pattern_id in enumerate(pattern_ids):
        background = backgrounds[pattern_id]
        for bin_id in range(bin_count):
            ax = axes[row_idx, bin_id]
            row = rows_by_key.get((pattern_id, bin_id))
            mean_map = maps_by_key.get((pattern_id, bin_id))
            if row is None or mean_map is None or int(row["sample_count"]) == 0:
                ax.axis("off")
                ax.set_title(f"B{bin_id}\nn=0", fontsize=7)
                continue
            norm_map = normalize(mean_map)
            tile = overlay_frame(background, norm_map, alpha=0.48)
            ax.imshow(tile)
            mask = cv2.resize(
                (norm_map >= primary_threshold).astype(np.float32),
                (background.shape[1], background.shape[0]),
                interpolation=0,
            )
            if np.any(mask):
                ax.contour(mask, levels=[0.5], colors="white", linewidths=0.6)
                ax.contour(mask, levels=[0.5], colors="black", linewidths=0.25)
            ax.set_title(
                f"B{bin_id} n={int(row['sample_count'])}\narea={threshold_area(row, primary_threshold):.1f}%",
                fontsize=6.5,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == len(pattern_ids) - 1:
                span = bin_ranges[bin_id]
                ax.set_xlabel(f"{span['min']:.2g}\n{span['max']:.2g}", fontsize=6.2)
        axes[row_idx, 0].set_ylabel(f"Pattern {pattern_id}", fontsize=9)
    fig.suptitle(f"CPU Grad-CAM grouped by {feature} rank bins", fontsize=13)
    fig.text(
        0.5,
        0.976,
        "Rows are real background patterns; columns are 10 equal-count rank bins. White contour is CAM >= 0.75.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_area_heatmap(
    *,
    output_path: Path,
    title: str,
    rows: list[dict[str, Any]],
    bin_count: int,
    field: str = "active_area_fraction_t0p75",
) -> None:
    pattern_ids = sorted({str(row["pattern_id"]) for row in rows}, key=metric_sort_key)
    matrix = np.full((len(pattern_ids), bin_count), np.nan, dtype=np.float32)
    rows_by_key = {(str(row["pattern_id"]), int(row["bin"])): row for row in rows}
    for row_idx, pattern_id in enumerate(pattern_ids):
        for bin_id in range(bin_count):
            row = rows_by_key.get((pattern_id, bin_id))
            if row is None:
                continue
            value = row.get(field)
            if value is not None:
                matrix[row_idx, bin_id] = float(value) * 100.0
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(bin_count))
    ax.set_xticklabels([str(idx) for idx in range(bin_count)])
    ax.set_yticks(np.arange(len(pattern_ids)))
    ax.set_yticklabels([str(pattern) for pattern in pattern_ids])
    ax.set_xlabel("bin")
    ax.set_ylabel("pattern")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("active area >= 0.75 (%)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def load_existing_viscosity_area(existing_csv: Path, output_dir: Path) -> list[dict[str, Any]]:
    with existing_csv.open("r", newline="") as file:
        source_rows = list(csv.DictReader(file))
    rows = []
    for source in source_rows:
        row = dict(source)
        row["axis"] = "viscosity_class"
        row["bin"] = int(source["true_viscosity_class"])
        rows.append(row)
    write_csv(output_dir / "viscosity_axis_activation_area_summary.csv", rows)
    plot_area_heatmap(
        output_path=output_dir / "viscosity_axis_active_area_t0p75_heatmap.png",
        title="Existing viscosity-axis Grad-CAM active area",
        rows=rows,
        bin_count=10,
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rebuild/retries/realonly_993_window30x21_no_rpm_ep50.yaml")
    parser.add_argument("--checkpoint", default="repro_realonly_993_window30x21_no_rpm_ep50.pth")
    parser.add_argument("--prediction-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--output-dir",
        default="outputs/rebuild_reproduction/gradcam_no_rpm_window30x21_dimensionless_bins_cpu",
    )
    parser.add_argument(
        "--existing-viscosity-area-csv",
        default="outputs/rebuild_reproduction/gradcam_no_rpm_window30x21_by_pattern/pattern_viscosity_gradcam_area_summary.csv",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--bin-count", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--source-frame-count", type=int, default=50)
    parser.add_argument("--target-layer", default="featureextractor.embeddings.patch_embeddings.projection")
    parser.add_argument("--gradcam-target", choices=["prediction", "true"], default="prediction")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--coverage-audit-only",
        action="store_true",
        help="Write selected-record coverage audit and exit before loading model/checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir = output_dir / "aggregate_maps"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config))
    if bool(config["model"]["embeddings"]["rpm_bool"]):
        raise ValueError("This diagnostic requires model.embeddings.rpm_bool: false")
    if int(config["model"]["transformer"].get("num_frames", 0)) != int(args.window_size):
        raise ValueError(f"Expected model num_frames={args.window_size}")

    prediction_dir = Path(args.prediction_dir)
    per_video = load_json(prediction_dir / "variable_window_predictions_mean_logits_per_video.json")
    per_window = load_json(prediction_dir / "variable_window_predictions_per_window.json")
    selected = [choose_window_record(video, per_window) for video in per_video]
    if args.max_samples is not None:
        selected = selected[: int(args.max_samples)]

    videos_by_name, params_by_name = split_dataset_paths(config)
    test_root = Path(config["dataset"]["test"]["test_root"])
    for record in selected:
        name = str(record["name"])
        record["pattern_id"] = load_pattern_id(params_by_name, name)
        record.update(load_raw_physics(test_root, name))

    bin_ranges_by_feature = {
        feature: assign_rank_bins(selected, feature, int(args.bin_count)) for feature in PHYSICS_FEATURES
    }
    pattern_ids = sorted({str(record["pattern_id"]) for record in selected}, key=metric_sort_key)
    coverage_guard = validate_full_run_coverage(
        selected_count=len(selected),
        pattern_ids=pattern_ids,
        max_samples=args.max_samples,
    )
    coverage_audit = write_coverage_audit(
        output_dir=output_dir,
        args=args,
        prediction_dir=prediction_dir,
        per_video_count=len(per_video),
        per_window_count=len(per_window),
        selected=selected,
        pattern_ids=pattern_ids,
        bin_ranges_by_feature=bin_ranges_by_feature,
        guard=coverage_guard,
    )
    if args.coverage_audit_only:
        print(
            json.dumps(
                {
                    "coverage_audit_json": str(output_dir / "coverage_audit.json"),
                    "coverage_audit_md": str(output_dir / "coverage_audit.md"),
                    "selected_count": len(selected),
                    "pattern_ids": pattern_ids,
                    "pattern_counts": coverage_audit["pattern_counts"],
                    "guard": coverage_guard,
                },
                indent=2,
            )
        )
        return 0

    model = make_model(config, device, output_attentions=False)
    checkpoint = resolve_checkpoint_path(config, args.checkpoint)
    load_info = load_state(model, checkpoint, device)
    model.eval()
    cam = Conv3DGradCam(model, args.target_layer)
    loader = WindowLoader(source_frame_count=args.source_frame_count, window_size=args.window_size)

    sums: dict[str, dict[tuple[str, int], np.ndarray]] = {feature: {} for feature in PHYSICS_FEATURES}
    counts: dict[str, dict[tuple[str, int], int]] = {feature: {} for feature in PHYSICS_FEATURES}
    correct_counts: dict[str, dict[tuple[str, int], int]] = {feature: {} for feature in PHYSICS_FEATURES}
    sample_rows: list[dict[str, Any]] = []
    try:
        for start_idx in range(0, len(selected), int(args.batch_size)):
            batch = selected[start_idx : start_idx + int(args.batch_size)]
            frames_batch = []
            for record in batch:
                frames, _visible = loader.load_window(videos_by_name[str(record["name"])], int(record["window_start"]))
                frames_batch.append(frames.squeeze(0))
            frames_tensor = torch.stack(frames_batch, dim=0).to(device)
            rpm_idx = torch.zeros((len(batch),), dtype=torch.long, device=device)
            pattern_tensor = torch.zeros((len(batch), 224, 224, 3), dtype=torch.float32, device=device)

            model.zero_grad(set_to_none=True)
            logits = model(frames_tensor, rpm_idx, pattern_tensor)
            predictions = logits.argmax(dim=1)
            true_classes = torch.tensor([int(record["true_viscosity_class"]) for record in batch], device=device)
            targets = predictions if args.gradcam_target == "prediction" else true_classes
            scores = logits[torch.arange(len(batch), device=device), targets].sum()
            scores.backward()
            volumes = cam.volumes()
            probs = F.softmax(logits.detach().cpu().float(), dim=1)

            for item_idx, (record, volume) in enumerate(zip(batch, volumes)):
                spatial = np.nanmean(volume, axis=2).astype(np.float32)
                pattern_id = str(record["pattern_id"])
                pred = int(predictions[item_idx].detach().cpu().item())
                true_class = int(record["true_viscosity_class"])
                correct = pred == true_class
                sample_row = {
                    "idx": int(start_idx + item_idx),
                    "name": str(record["name"]),
                    "pattern_id": pattern_id,
                    "true_viscosity_class": true_class,
                    "prediction": pred,
                    "correct": correct,
                    "window_start": int(record["window_start"]),
                    "confidence": float(probs[item_idx, pred].item()),
                    "spatial_cam_mean": float(np.mean(spatial)),
                    "spatial_cam_max": float(np.max(spatial)),
                }
                for feature in PHYSICS_FEATURES:
                    bin_id = int(record[f"{feature}_bin"])
                    key = (pattern_id, bin_id)
                    sums[feature][key] = sums[feature].get(key, np.zeros_like(spatial)) + spatial
                    counts[feature][key] = counts[feature].get(key, 0) + 1
                    correct_counts[feature][key] = correct_counts[feature].get(key, 0) + int(correct)
                    sample_row[feature] = float(record[feature])
                    sample_row[f"log10_{feature}"] = float(record[f"log10_{feature}"])
                    sample_row[f"{feature}_bin"] = bin_id
                sample_rows.append(sample_row)
    finally:
        cam.close()

    backgrounds = {pattern_id: load_background_image(test_root, pattern_id) for pattern_id in pattern_ids}
    all_area_rows: list[dict[str, Any]] = []
    feature_outputs: dict[str, Any] = {}
    for feature in PHYSICS_FEATURES:
        feature_maps: dict[tuple[str, int], np.ndarray] = {}
        feature_rows: list[dict[str, Any]] = []
        for pattern_id in pattern_ids:
            for bin_id in range(int(args.bin_count)):
                key = (pattern_id, bin_id)
                count = counts[feature].get(key, 0)
                if count:
                    mean_map = sums[feature][key] / count
                    feature_maps[key] = mean_map
                    np.save(
                        aggregate_dir / f"{feature}_pattern_{safe_name(pattern_id)}_bin_{bin_id}_mean_spatial_gradcam.npy",
                        mean_map.astype(np.float32),
                    )
                    row = active_area_metrics(pattern_id, mean_map, count)
                else:
                    row = {
                        "pattern_id": pattern_id,
                        "sample_count": 0,
                        "grid_height": None,
                        "grid_width": None,
                        "threshold_basis": "empty bin",
                        "primary_threshold": 0.75,
                        "mean_cam": None,
                        "max_cam": None,
                    }
                    for threshold in (0.25, 0.50, 0.75):
                        suffix = str(threshold).replace(".", "p")
                        row[f"active_tokens_t{suffix}"] = None
                        row[f"active_area_fraction_t{suffix}"] = None
                        row[f"equivalent_image_pixels_t{suffix}"] = None
                        row[f"cam_mass_share_t{suffix}"] = None
                row["axis"] = feature
                row["bin"] = bin_id
                row["correct_count"] = correct_counts[feature].get(key, 0)
                row["accuracy"] = (
                    correct_counts[feature].get(key, 0) / count if count else None
                )
                bin_span = bin_ranges_by_feature[feature][bin_id]
                row["bin_min"] = bin_span["min"]
                row["bin_max"] = bin_span["max"]
                row["bin_mean"] = bin_span["mean"]
                feature_rows.append(row)
                all_area_rows.append(row)
        feature_csv = output_dir / f"{feature}_pattern_bin_activation_area_summary.csv"
        write_csv(feature_csv, feature_rows)
        write_json(output_dir / f"{feature}_pattern_bin_activation_area_summary.json", feature_rows)
        grid_path = output_dir / f"{feature}_pattern_bin_gradcam_grid.png"
        plot_feature_grid(
            feature=feature,
            output_path=grid_path,
            rows=feature_rows,
            maps_by_key=feature_maps,
            backgrounds=backgrounds,
            bin_ranges=bin_ranges_by_feature[feature],
            bin_count=int(args.bin_count),
            primary_threshold=0.75,
        )
        heatmap_path = output_dir / f"{feature}_active_area_t0p75_heatmap.png"
        plot_area_heatmap(
            output_path=heatmap_path,
            title=f"{feature}: active area by pattern and rank bin",
            rows=feature_rows,
            bin_count=int(args.bin_count),
        )
        feature_outputs[feature] = {
            "area_csv": str(feature_csv),
            "grid": str(grid_path),
            "heatmap": str(heatmap_path),
            "bin_ranges": bin_ranges_by_feature[feature],
        }

    write_csv(output_dir / "all_dimensionless_axes_activation_area_summary.csv", all_area_rows)
    write_csv(output_dir / "dimensionless_gradcam_sample_metrics.csv", sample_rows)
    write_json(output_dir / "dimensionless_gradcam_sample_metrics.json", sample_rows)
    viscosity_rows = load_existing_viscosity_area(Path(args.existing_viscosity_area_csv), output_dir)

    summary = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "load_info": load_info,
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "sample_count": len(sample_rows),
        "pattern_ids": pattern_ids,
        "pattern_counts": coverage_audit["pattern_counts"],
        "bin_count": int(args.bin_count),
        "features": feature_outputs,
        "coverage_audit_json": str(output_dir / "coverage_audit.json"),
        "coverage_audit_md": str(output_dir / "coverage_audit.md"),
        "viscosity_axis_area_csv": str(output_dir / "viscosity_axis_activation_area_summary.csv"),
        "viscosity_axis_area_rows": len(viscosity_rows),
        "existing_viscosity_area_csv": args.existing_viscosity_area_csv,
        "target_layer": args.target_layer,
        "gradcam_target": args.gradcam_target,
        "device": "cpu",
        "torch_threads": int(args.torch_threads),
        "threshold_basis": "per-pattern/bin accumulated mean Grad-CAM min-max normalized to [0,1]",
        "primary_threshold": 0.75,
    }
    write_json(output_dir / "summary.json", summary)
    report_lines = [
        "# CPU Grad-CAM Dimensionless Bins",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Config: `{args.config}`",
        f"- Samples: {len(sample_rows)}",
        f"- Device: CPU",
        f"- Pattern rows: `{', '.join(pattern_ids)}`",
        f"- Pattern counts: `{coverage_audit['pattern_counts']}`",
        f"- Coverage audit: `{output_dir / 'coverage_audit.json'}`",
        "- Bin policy: 10 equal-count rank bins per physics proxy over selected real-test samples.",
        "- Activation area is measured after per-group accumulated mean Grad-CAM min-max normalization.",
        "",
        "## Outputs",
        "",
    ]
    for feature, paths in feature_outputs.items():
        report_lines.append(f"- {feature} grid: `{paths['grid']}`")
        report_lines.append(f"- {feature} area CSV: `{paths['area_csv']}`")
    report_lines.append(f"- Viscosity-axis area CSV: `{summary['viscosity_axis_area_csv']}`")
    report_lines.append(f"- Summary: `{output_dir / 'summary.json'}`")
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
