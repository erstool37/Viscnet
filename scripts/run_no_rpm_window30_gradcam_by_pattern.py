#!/usr/bin/env python3
"""Accumulate no-RPM sliding-window Grad-CAM maps by real background pattern."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
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
from scripts.run_allnew_no_rpm_diagnostics import load_state, make_model, resolve_checkpoint  # noqa: E402


THRESHOLDS = (0.25, 0.50, 0.75)


def load_pattern_id(params_by_name: dict[str, str], name: str) -> str:
    with Path(params_by_name[name]).open("r") as file:
        payload = json.load(file)
    return str(payload.get("background", infer_pattern_from_render(name)))


def infer_pattern_from_render(name: str) -> str:
    render = name.split("_render")[-1]
    if render in set("ABCDE"):
        return "1"
    if render in set("KLMNO"):
        return "2"
    if render in set("PQRST"):
        return "3"
    if render in set("UVWXY"):
        return "4"
    return "unknown"


def load_background_image(test_root: Path, pattern_id: str, image_size: int = 224) -> np.ndarray:
    path = test_root / "backgrounds" / f"{pattern_id}.png"
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


def active_area_metrics(
    pattern_id: str,
    mean_map: np.ndarray,
    sample_count: int,
    true_viscosity_class: int | None = None,
) -> dict[str, Any]:
    norm_map = normalize(mean_map)
    total_tokens = int(norm_map.size)
    row: dict[str, Any] = {
        "pattern_id": str(pattern_id),
        "sample_count": int(sample_count),
        "grid_height": int(norm_map.shape[0]),
        "grid_width": int(norm_map.shape[1]),
        "threshold_basis": "per-group accumulated mean Grad-CAM min-max normalized to [0,1]",
        "primary_threshold": 0.75,
        "mean_cam": float(np.mean(norm_map)),
        "max_cam": float(np.max(norm_map)),
    }
    if true_viscosity_class is not None:
        row["true_viscosity_class"] = int(true_viscosity_class)
    for threshold in THRESHOLDS:
        mask = norm_map >= threshold
        active_tokens = int(mask.sum())
        area_fraction = active_tokens / total_tokens if total_tokens else 0.0
        mass = float(norm_map[mask].sum() / (norm_map.sum() + 1e-8)) if active_tokens else 0.0
        suffix = str(threshold).replace(".", "p")
        row[f"active_tokens_t{suffix}"] = active_tokens
        row[f"active_area_fraction_t{suffix}"] = float(area_fraction)
        row[f"equivalent_image_pixels_t{suffix}"] = int(round(area_fraction * 224 * 224))
        row[f"cam_mass_share_t{suffix}"] = mass
    return row


def threshold_suffix(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def threshold_area(row: dict[str, Any], threshold: float) -> float:
    return 100.0 * float(row[f"active_area_fraction_t{threshold_suffix(threshold)}"])


def threshold_mass(row: dict[str, Any], threshold: float) -> float:
    return 100.0 * float(row[f"cam_mass_share_t{threshold_suffix(threshold)}"])


def threshold_mask_overlay(background: np.ndarray, mean_map: np.ndarray, threshold: float) -> np.ndarray:
    mask = mean_map >= float(threshold)
    mask_big = cv2.resize(mask.astype(np.float32), (background.shape[1], background.shape[0]), interpolation=0)
    mask_rgb = np.zeros_like(background)
    mask_rgb[..., 0] = 1.0
    return np.clip(0.55 * background + 0.45 * mask_rgb * mask_big[..., None], 0.0, 1.0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_pattern_stack(
    pattern_rows: list[dict[str, Any]],
    output_path: Path,
    primary_threshold: float = 0.75,
) -> None:
    fig, axes = plt.subplots(len(pattern_rows), 4, figsize=(13.0, 3.1 * len(pattern_rows)), squeeze=False)
    for row_index, row in enumerate(pattern_rows):
        pattern_id = row["pattern_id"]
        background = row["background"]
        mean_map = normalize(row["mean_map"])
        overlay = overlay_frame(background, mean_map, alpha=0.48)
        masked_overlay = threshold_mask_overlay(background, mean_map, primary_threshold)

        titles = [
            f"Pattern {pattern_id}\nclean background",
            f"Pattern {pattern_id}\nmean Grad-CAM n={row['sample_count']}",
            f"Pattern {pattern_id}\noverlay",
            "Pattern {pattern}\nmask >= {threshold:.2f}, area={area:.1f}%, mass={mass:.1f}%".format(
                pattern=pattern_id,
                threshold=primary_threshold,
                area=threshold_area(row, primary_threshold),
                mass=threshold_mass(row, primary_threshold),
            ),
        ]
        images = [background, mean_map, overlay, masked_overlay]
        cmaps = [None, "jet", None, None]
        for col_index, (title, image, cmap) in enumerate(zip(titles, images, cmaps)):
            ax = axes[row_index, col_index]
            if cmap:
                ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
            else:
                ax.imshow(image)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Accumulated Grad-CAM by Real Background Pattern", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pattern_viscosity_four_panel_grid(
    pattern_class_rows: list[dict[str, Any]],
    output_path: Path,
    pattern_ids: list[str],
    class_count: int,
) -> None:
    by_key = {
        (str(row["pattern_id"]), int(row["true_viscosity_class"])): row
        for row in pattern_class_rows
    }
    panel_specs: list[tuple[str, float | None]] = [
        ("CAM overlay", None),
        ("Mask >= 0.25", 0.25),
        ("Mask >= 0.50", 0.50),
        ("Mask >= 0.75", 0.75),
    ]
    rows = len(panel_specs) * len(pattern_ids)
    fig, axes = plt.subplots(rows, class_count, figsize=(2.05 * class_count, 1.95 * rows), squeeze=False)

    for panel_index, (panel_name, threshold) in enumerate(panel_specs):
        for pattern_index, pattern_id in enumerate(pattern_ids):
            axis_row = panel_index * len(pattern_ids) + pattern_index
            for true_class in range(class_count):
                ax = axes[axis_row, true_class]
                row = by_key.get((str(pattern_id), true_class))
                if row is None or int(row["sample_count"]) == 0:
                    ax.axis("off")
                    ax.set_title(f"C{true_class}\nn=0", fontsize=7)
                    continue
                background = row["background"]
                mean_map = normalize(row["mean_map"])
                if threshold is None:
                    tile = overlay_frame(background, mean_map, alpha=0.48)
                    title = f"C{true_class}\nn={int(row['sample_count'])}"
                else:
                    tile = threshold_mask_overlay(background, mean_map, threshold)
                    title = (
                        f"C{true_class} n={int(row['sample_count'])}\n"
                        f"A={threshold_area(row, threshold):.1f}% M={threshold_mass(row, threshold):.1f}%"
                    )
                ax.imshow(tile)
                ax.set_title(title, fontsize=6.5)
                ax.set_xticks([])
                ax.set_yticks([])
            axes[axis_row, 0].set_ylabel(f"{panel_name}\nPattern {pattern_id}", fontsize=9)

    fig.suptitle("Accumulated Grad-CAM by Pattern and True Viscosity Class", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pattern_viscosity_grid(
    pattern_class_rows: list[dict[str, Any]],
    output_path: Path,
    pattern_ids: list[str],
    class_count: int,
    primary_threshold: float = 0.75,
) -> None:
    by_key = {
        (str(row["pattern_id"]), int(row["true_viscosity_class"])): row
        for row in pattern_class_rows
    }
    fig, axes = plt.subplots(
        len(pattern_ids),
        class_count,
        figsize=(2.0 * class_count, 2.15 * len(pattern_ids)),
        squeeze=False,
    )
    for row_index, pattern_id in enumerate(pattern_ids):
        for true_class in range(class_count):
            ax = axes[row_index, true_class]
            row = by_key.get((str(pattern_id), true_class))
            if row is None or int(row["sample_count"]) == 0:
                ax.axis("off")
                ax.set_title(f"P{pattern_id} C{true_class}\nn=0", fontsize=7)
                continue
            background = row["background"]
            mean_map = normalize(row["mean_map"])
            tile = overlay_frame(background, mean_map, alpha=0.48)
            ax.imshow(tile)
            mask = cv2.resize(
                (mean_map >= primary_threshold).astype(np.float32),
                (background.shape[1], background.shape[0]),
                interpolation=0,
            )
            if np.any(mask):
                ax.contour(mask, levels=[0.5], colors="red", linewidths=0.7)
            ax.set_title(
                "P{pattern} C{cls}\nn={n} area={area:.1f}%\nmass={mass:.1f}%".format(
                    pattern=pattern_id,
                    cls=true_class,
                    n=int(row["sample_count"]),
                area=threshold_area(row, primary_threshold),
                mass=threshold_mass(row, primary_threshold),
                ),
                fontsize=7,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_index, 0].set_ylabel(f"Pattern {pattern_id}", fontsize=10)
    fig.suptitle("Accumulated Grad-CAM Overlays by Pattern and True Viscosity Class", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Grad-CAM By Pattern",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Config: `{summary['config']}`",
        f"- Samples accumulated: {summary['sample_count']}",
        f"- Runtime RPM tensor nonzero count: {summary['rpm_model_nonzero_count']}",
        f"- Pattern stack image: `{summary['pattern_stack_path']}`",
        f"- Pattern x viscosity grid: `{summary['pattern_viscosity_grid_path']}`",
        f"- Four-panel threshold grid: `{summary['pattern_viscosity_four_panel_path']}`",
        "",
        "## Thresholding",
        "",
        "- Per-sample Grad-CAM volumes are ReLU Grad-CAM maps from the ViViT tubelet Conv3d projection.",
        "- Each per-sample volume is min-max normalized to `[0,1]` after Grad-CAM.",
        "- A per-sample spatial map is the temporal mean across the 15 tubelet time steps.",
        "- For each pattern and each pattern/viscosity-class cell, all matching per-sample spatial maps are averaged.",
        "- For visualization and area measurement, each accumulated group map is min-max normalized again to `[0,1]`.",
        "- Final panels show continuous CAM overlay plus masks at `>= 0.25`, `>= 0.50`, and `>= 0.75`.",
        "- Main active area threshold is `>= 0.75`; lower thresholds `>= 0.25` and `>= 0.50` show broader sensitivity.",
        "",
        "## Area By Pattern",
        "",
        "| pattern | n | area >=0.25 | area >=0.50 | area >=0.75 | tokens >=0.75 | mass >=0.75 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {pattern} | {n} | {a25:.3f} | {a50:.3f} | {a75:.3f} | {tok75} | {mass75:.3f} |".format(
                pattern=row["pattern_id"],
                n=row["sample_count"],
                a25=row["active_area_fraction_t0p25"],
                a50=row["active_area_fraction_t0p5"],
                a75=row["active_area_fraction_t0p75"],
                tok75=row["active_tokens_t0p75"],
                mass75=row["cam_mass_share_t0p75"],
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Metrics CSV: `{output_dir / 'pattern_gradcam_area_summary.csv'}`",
            f"- Metrics JSON: `{output_dir / 'pattern_gradcam_area_summary.json'}`",
            f"- Pattern x viscosity metrics CSV: `{output_dir / 'pattern_viscosity_gradcam_area_summary.csv'}`",
            f"- Pattern x viscosity metrics JSON: `{output_dir / 'pattern_viscosity_gradcam_area_summary.json'}`",
            f"- Per-sample CSV: `{output_dir / 'pattern_gradcam_sample_metrics.csv'}`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rebuild/retries/realonly_993_window30x21_no_rpm_ep50.yaml")
    parser.add_argument("--checkpoint", default="repro_realonly_993_window30x21_no_rpm_ep50.pth")
    parser.add_argument("--prediction-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", default="outputs/rebuild_reproduction/gradcam_no_rpm_window30x21_by_pattern")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--source-frame-count", type=int, default=50)
    parser.add_argument("--target-layer", default="featureextractor.embeddings.patch_embeddings.projection")
    parser.add_argument("--gradcam-target", choices=["prediction", "true"], default="prediction")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    if bool(config["model"]["embeddings"]["rpm_bool"]):
        raise ValueError("This diagnostic requires model.embeddings.rpm_bool: false")
    if int(config["model"]["transformer"].get("num_frames", 0)) != int(args.window_size):
        raise ValueError(f"Expected model num_frames={args.window_size}")

    prediction_dir = Path(args.prediction_dir)
    per_video = load_json(prediction_dir / "variable_window_predictions_mean_logits_per_video.json")
    per_window = load_json(prediction_dir / "variable_window_predictions_per_window.json")
    selected = [choose_window_record(video, per_window) for video in per_video]

    videos_by_name, params_by_name = split_dataset_paths(config)
    for record in selected:
        record["pattern_id"] = load_pattern_id(params_by_name, str(record["name"]))

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = make_model(config, device, output_attentions=False)
    checkpoint = resolve_checkpoint(config, args.checkpoint)
    load_info = load_state(model, checkpoint, device)
    model.eval()
    cam = Conv3DGradCam(model, args.target_layer)
    loader = WindowLoader(source_frame_count=args.source_frame_count, window_size=args.window_size)

    pattern_sums: dict[str, np.ndarray] = {}
    pattern_counts: dict[str, int] = {}
    pattern_class_sums: dict[tuple[str, int], np.ndarray] = {}
    pattern_class_counts: dict[tuple[str, int], int] = {}
    sample_rows: list[dict[str, Any]] = []
    rpm_model_nonzero = 0
    try:
        for start_idx in range(0, len(selected), int(args.batch_size)):
            batch = selected[start_idx : start_idx + int(args.batch_size)]
            frame_tensors = []
            for record in batch:
                frames, _visible = loader.load_window(videos_by_name[str(record["name"])], int(record["window_start"]))
                frame_tensors.append(frames.squeeze(0))
            frames_batch = torch.stack(frame_tensors, dim=0).to(device)
            rpm_idx = torch.zeros((len(batch),), dtype=torch.long, device=device)
            pattern = torch.zeros((len(batch), 224, 224, 3), dtype=torch.float32, device=device)
            rpm_model_nonzero += int((rpm_idx.detach().cpu() != 0).sum().item())

            model.zero_grad(set_to_none=True)
            logits = model(frames_batch, rpm_idx, pattern)
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
                true_class = int(record["true_viscosity_class"])
                pattern_class_key = (pattern_id, true_class)
                pattern_sums[pattern_id] = pattern_sums.get(pattern_id, np.zeros_like(spatial)) + spatial
                pattern_counts[pattern_id] = pattern_counts.get(pattern_id, 0) + 1
                pattern_class_sums[pattern_class_key] = (
                    pattern_class_sums.get(pattern_class_key, np.zeros_like(spatial)) + spatial
                )
                pattern_class_counts[pattern_class_key] = pattern_class_counts.get(pattern_class_key, 0) + 1
                pred = int(predictions[item_idx].detach().cpu().item())
                sample_rows.append(
                    {
                        "idx": int(start_idx + item_idx),
                        "name": str(record["name"]),
                        "pattern_id": pattern_id,
                        "video_idx": int(record["video_idx"]),
                        "sample_idx": int(record["sample_idx"]),
                        "window_start": int(record["window_start"]),
                        "true_viscosity_class": true_class,
                        "prediction": pred,
                        "correct": pred == true_class,
                        "target_class": int(targets[item_idx].detach().cpu().item()),
                        "confidence": float(probs[item_idx, pred].item()),
                        "spatial_cam_mean": float(np.mean(spatial)),
                        "spatial_cam_max": float(np.max(spatial)),
                    }
                )
    finally:
        cam.close()

    test_root = Path(config["dataset"]["test"]["test_root"])
    class_count = int(config["model"]["transformer"]["class"])
    rows: list[dict[str, Any]] = []
    pattern_class_rows: list[dict[str, Any]] = []
    pattern_panel_rows: list[dict[str, Any]] = []
    aggregate_dir = output_dir / "aggregate_maps"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    pattern_ids = sorted(pattern_sums, key=lambda value: int(value) if value.isdigit() else value)
    backgrounds = {pattern_id: load_background_image(test_root, pattern_id) for pattern_id in pattern_ids}
    for pattern_id in pattern_ids:
        mean_map = pattern_sums[pattern_id] / max(1, pattern_counts[pattern_id])
        np.save(aggregate_dir / f"pattern_{safe_name(pattern_id)}_mean_spatial_gradcam.npy", mean_map.astype(np.float32))
        row = active_area_metrics(pattern_id, mean_map, pattern_counts[pattern_id])
        rows.append(row)
        panel_row = dict(row)
        panel_row["mean_map"] = mean_map
        panel_row["background"] = backgrounds[pattern_id]
        pattern_panel_rows.append(panel_row)

    for pattern_id in pattern_ids:
        for true_class in range(class_count):
            key = (pattern_id, true_class)
            if key not in pattern_class_sums:
                continue
            mean_map = pattern_class_sums[key] / max(1, pattern_class_counts[key])
            np.save(
                aggregate_dir / f"pattern_{safe_name(pattern_id)}_class_{true_class}_mean_spatial_gradcam.npy",
                mean_map.astype(np.float32),
            )
            row = active_area_metrics(
                pattern_id,
                mean_map,
                pattern_class_counts[key],
                true_viscosity_class=true_class,
            )
            row["mean_map"] = mean_map
            row["background"] = backgrounds[pattern_id]
            pattern_class_rows.append(row)

    pattern_stack_path = output_dir / "all_patterns_accumulated_gradcam_stack.png"
    plot_pattern_stack(pattern_panel_rows, pattern_stack_path)
    pattern_viscosity_grid_path = output_dir / "all_patterns_by_viscosity_accumulated_gradcam_grid.png"
    plot_pattern_viscosity_grid(
        pattern_class_rows,
        pattern_viscosity_grid_path,
        pattern_ids=pattern_ids,
        class_count=class_count,
    )
    pattern_viscosity_four_panel_path = output_dir / "all_patterns_by_viscosity_four_panel_thresholds.png"
    plot_pattern_viscosity_four_panel_grid(
        pattern_class_rows,
        pattern_viscosity_four_panel_path,
        pattern_ids=pattern_ids,
        class_count=class_count,
    )
    serializable_pattern_class_rows = [
        {key: value for key, value in row.items() if key not in {"mean_map", "background"}}
        for row in pattern_class_rows
    ]
    write_csv(output_dir / "pattern_gradcam_area_summary.csv", rows)
    write_csv(output_dir / "pattern_viscosity_gradcam_area_summary.csv", serializable_pattern_class_rows)
    write_csv(output_dir / "pattern_gradcam_sample_metrics.csv", sample_rows)
    (output_dir / "pattern_gradcam_area_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output_dir / "pattern_viscosity_gradcam_area_summary.json").write_text(
        json.dumps(serializable_pattern_class_rows, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "pattern_gradcam_sample_metrics.json").write_text(
        json.dumps(sample_rows, indent=2) + "\n", encoding="utf-8"
    )
    pattern_class_counts_summary = {
        f"pattern_{pattern_id}_class_{true_class}": pattern_class_counts[(pattern_id, true_class)]
        for pattern_id, true_class in sorted(
            pattern_class_counts,
            key=lambda item: (int(item[0]) if str(item[0]).isdigit() else str(item[0]), int(item[1])),
        )
    }
    summary = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "load_info": load_info,
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "sample_count": len(sample_rows),
        "pattern_counts": {pattern_id: pattern_counts[pattern_id] for pattern_id in sorted(pattern_counts)},
        "pattern_class_counts": pattern_class_counts_summary,
        "thresholds": list(THRESHOLDS),
        "primary_threshold": 0.75,
        "threshold_basis": "per-group accumulated mean Grad-CAM min-max normalized to [0,1]",
        "pattern_stack_path": str(pattern_stack_path),
        "pattern_viscosity_grid_path": str(pattern_viscosity_grid_path),
        "pattern_viscosity_four_panel_path": str(pattern_viscosity_four_panel_path),
        "rpm_bool": False,
        "rpm_model_nonzero_count": rpm_model_nonzero,
        "target_layer": args.target_layer,
        "gradcam_target": args.gradcam_target,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir, summary, rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
