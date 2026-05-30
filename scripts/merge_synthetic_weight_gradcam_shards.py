#!/usr/bin/env python3
"""Merge sharded synthetic-weight Grad-CAM outputs."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_synthetic_weight_gradcam_bins import (  # noqa: E402
    THRESHOLDS,
    active_area_metrics,
    load_background,
    load_config,
    plot_area_heatmap,
    plot_pattern_bin_pages,
    safe_name,
    split_dataset_paths,
    write_csv,
)


def read_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def sorted_pattern_ids(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: int(value) if str(value).isdigit() else str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["real_test", "synthetic_val"], required=True)
    parser.add_argument(
        "--output-root",
        default="outputs/rebuild_reproduction/synthetic_weight_gradcam_bins/allnew_synth_no_rpm_augv1_ep80",
    )
    parser.add_argument("--patterns-per-page", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_root) / args.split
    shard_root = output_dir / "shards"
    shard_dirs = sorted(path.parent for path in shard_root.glob("shard_*_of_*/summary.json"))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard summaries found under {shard_root}")

    shard_summaries = [read_json(path / "summary.json") for path in shard_dirs]
    first_summary = shard_summaries[0]
    split = str(first_summary["split"])
    if split != args.split:
        raise ValueError(f"Shard split {split} does not match requested split {args.split}")

    config_path = Path(first_summary["config"])
    config = load_config(config_path)
    _video_paths, _para_paths, _section, _dataset_name, dataset_root = split_dataset_paths(config, args.split)
    class_count = int(first_summary["class_count"])
    image_size = int(config["model"]["transformer"].get("image_size", 224))

    group_sums: dict[tuple[str, int], np.ndarray] = {}
    group_counts: dict[tuple[str, int], int] = {}
    group_correct: dict[tuple[str, int], int] = {}
    sample_rows: list[dict[str, Any]] = []
    pattern_ids_seen: set[str] = set()

    for shard_dir, shard_summary in zip(shard_dirs, shard_summaries):
        area_rows = read_json(shard_dir / f"{args.split}_pattern_bin_gradcam_area_summary.json")
        shard_samples = read_json(shard_dir / f"{args.split}_gradcam_sample_metrics.json")
        sample_rows.extend(shard_samples)
        for row in area_rows:
            pattern_id = str(row["pattern_id"])
            viscosity_bin = int(row["true_viscosity_bin"])
            pattern_ids_seen.add(pattern_id)
            count = int(row["sample_count"] or 0)
            if count <= 0:
                continue
            map_path = (
                shard_dir
                / "aggregate_maps"
                / f"{args.split}_pattern_{safe_name(pattern_id)}_bin_{viscosity_bin}_mean_spatial_gradcam.npy"
            )
            if not map_path.exists():
                raise FileNotFoundError(map_path)
            mean_map = np.load(map_path).astype(np.float32)
            key = (pattern_id, viscosity_bin)
            group_sums[key] = group_sums.get(key, np.zeros_like(mean_map)) + mean_map * count
            group_counts[key] = group_counts.get(key, 0) + count
            group_correct[key] = group_correct.get(key, 0) + int(row.get("correct_count") or 0)

    sample_rows = sorted(sample_rows, key=lambda row: int(row["idx"]))
    pattern_ids = sorted_pattern_ids(pattern_ids_seen)
    aggregate_dir = output_dir / "aggregate_maps"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

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
                metrics = active_area_metrics(mean_map, count)
            else:
                metrics = active_area_metrics(np.empty((0, 0)), 0)
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
    backgrounds = {pattern_id: load_background(dataset_root, pattern_id, image_size) for pattern_id in pattern_ids}
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
        "config": first_summary["config"],
        "checkpoint": first_summary["checkpoint"],
        "load_info": first_summary.get("load_info"),
        "sample_count": len(sample_rows),
        "total_samples_selected": first_summary.get("total_samples_selected"),
        "accuracy": accuracy,
        "pattern_count": len(pattern_ids),
        "pattern_ids": pattern_ids,
        "class_count": class_count,
        "thresholds": list(THRESHOLDS),
        "primary_threshold": 0.75,
        "threshold_basis": "weighted merge of per-shard per-pattern/bin accumulated mean Grad-CAM maps",
        "target_layer": first_summary.get("target_layer"),
        "gradcam_target": first_summary.get("gradcam_target"),
        "num_shards": len(shard_dirs),
        "shard_dirs": [str(path) for path in shard_dirs],
        "rpm_bool": first_summary.get("rpm_bool"),
        "rpm_input_policy": first_summary.get("rpm_input_policy"),
        "rpm_model_nonzero_count": int(sum(int(summary.get("rpm_model_nonzero_count", 0)) for summary in shard_summaries)),
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
        f"- Config: `{summary['config']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Split: `{args.split}`",
        f"- Samples: {summary['sample_count']}",
        f"- Accuracy: {summary['accuracy']}",
        f"- Patterns: {summary['pattern_count']}",
        f"- Viscosity-bin axis: true bins `0..{class_count - 1}`",
        f"- Merged shards: {len(shard_dirs)}",
        f"- Grad-CAM target: `{summary['gradcam_target']}` class score",
        f"- Target layer: `{summary['target_layer']}`",
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
