#!/usr/bin/env python3
"""Plot attention maps by true class, split into correct and wrong predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)


def load_mean_volume(records: list[dict[str, Any]], volume_dir: Path) -> np.ndarray | None:
    volumes = []
    for record in records:
        path = volume_dir / record["volume_file"]
        if path.exists():
            volume = np.load(path).astype(np.float32)
            if volume.ndim == 3 and volume.shape[2] > 0:
                volumes.append(volume)
    if not volumes:
        return None

    max_h = max(volume.shape[0] for volume in volumes)
    max_w = max(volume.shape[1] for volume in volumes)
    max_t = max(volume.shape[2] for volume in volumes)
    padded = np.full((len(volumes), max_h, max_w, max_t), np.nan, dtype=np.float32)
    for index, volume in enumerate(volumes):
        padded[index, : volume.shape[0], : volume.shape[1], : volume.shape[2]] = volume
    return np.nanmean(padded, axis=0)


def spatial_map(volume: np.ndarray) -> np.ndarray:
    return normalize(np.nanmean(normalize(volume), axis=2))


def temporal_curve(volume: np.ndarray) -> np.ndarray:
    return normalize(np.nanmean(normalize(volume), axis=(0, 1)))


def prediction_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    preds = [int(record["prediction"]) for record in records]
    return {str(label): preds.count(label) for label in sorted(set(preds))}


def summarize_groups(records: list[dict[str, Any]], class_labels: list[int]) -> list[dict[str, Any]]:
    rows = []
    for true_class in class_labels:
        class_records = [record for record in records if int(record["true_viscosity_class"]) == true_class]
        for correctness in [True, False]:
            group = [record for record in class_records if bool(record["correct"]) is correctness]
            rows.append(
                {
                    "true_viscosity_class": true_class,
                    "correct": correctness,
                    "count": len(group),
                    "prediction_counts": prediction_counts(group),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["true_viscosity_class", "correct", "count", "prediction_counts"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_spatial_grid(
    records: list[dict[str, Any]],
    volume_dir: Path,
    output_path: Path,
    class_labels: list[int],
) -> None:
    fig, axes = plt.subplots(2, len(class_labels), figsize=(2.4 * len(class_labels), 5.2), squeeze=False)
    row_specs = [("Correct", True), ("Wrong", False)]
    for row_index, (row_label, correctness) in enumerate(row_specs):
        for col_index, true_class in enumerate(class_labels):
            ax = axes[row_index, col_index]
            group = [
                record
                for record in records
                if int(record["true_viscosity_class"]) == true_class and bool(record["correct"]) is correctness
            ]
            mean_volume = load_mean_volume(group, volume_dir)
            if mean_volume is None:
                ax.axis("off")
                ax.set_title(f"class {true_class}\nn=0", fontsize=8)
                continue
            ax.imshow(spatial_map(mean_volume), cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
            pred_counts = prediction_counts(group)
            pred_text = ",".join(f"{label}:{count}" for label, count in pred_counts.items())
            ax.set_title(f"class {true_class}\nn={len(group)}\npred {pred_text}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_index, 0].set_ylabel(row_label, fontsize=11)
    fig.suptitle("Real Test Attention Maps by True Class and Correctness")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_temporal_grid(
    records: list[dict[str, Any]],
    volume_dir: Path,
    output_path: Path,
    class_labels: list[int],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), squeeze=False)
    for ax, title, correctness in [
        (axes[0, 0], "Correct", True),
        (axes[0, 1], "Wrong", False),
    ]:
        for true_class in class_labels:
            group = [
                record
                for record in records
                if int(record["true_viscosity_class"]) == true_class and bool(record["correct"]) is correctness
            ]
            mean_volume = load_mean_volume(group, volume_dir)
            if mean_volume is None:
                continue
            ax.plot(temporal_curve(mean_volume), marker="o", lw=1.1, ms=2.5, label=f"class {true_class} n={len(group)}")
        ax.set_title(title)
        ax.set_xlabel("tubelet time")
        ax.set_ylabel("attention")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Real Test Temporal Attention by True Class and Correctness")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dir", required=True, help="Directory containing real_test_attention_metrics.json")
    parser.add_argument("--class-count", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    attention_dir = Path(args.attention_dir)
    metrics_path = attention_dir / "real_test_attention_metrics.json"
    volume_dir = attention_dir / "volumes"
    panel_dir = attention_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("r") as file:
        records = json.load(file)

    class_labels = list(range(args.class_count))
    summary = summarize_groups(records, class_labels)
    summary_json = attention_dir / "attention_true_class_correct_wrong_summary.json"
    summary_csv = attention_dir / "attention_true_class_correct_wrong_summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(summary_csv, summary)

    spatial_path = panel_dir / "attention_by_true_class_correct_wrong_spatial.png"
    temporal_path = panel_dir / "attention_by_true_class_correct_wrong_temporal.png"
    plot_spatial_grid(records, volume_dir, spatial_path, class_labels)
    plot_temporal_grid(records, volume_dir, temporal_path, class_labels)

    print(f"Wrote {spatial_path}")
    print(f"Wrote {temporal_path}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
