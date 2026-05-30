#!/usr/bin/env python3
"""Build a compact Grad-CAM overlay plus entropy/area summary panel."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_GRADCAM_DIR = Path("outputs/rebuild_reproduction/gradcam_no_rpm_window30x21_by_pattern")
DEFAULT_ENTROPY_CSV = Path(
    "outputs/rebuild_reproduction/entropy_probe/"
    "repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless/"
    "pattern_entropy_accuracy/pattern_entropy_accuracy_summary.csv"
)
DEFAULT_BACKGROUND_DIR = Path("dataset/RealArchive/test_1000_wo_pat2/backgrounds")
DEFAULT_OUTPUT = DEFAULT_GRADCAM_DIR / "all_patterns_by_viscosity_cam_overlay_entropy_area_panel.png"


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)


def overlay_frame(frame_rgb: np.ndarray, heatmap_small: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    heat = cv2.resize(heatmap_small, (frame_rgb.shape[1], frame_rgb.shape[0]))
    heat = normalize(heat)
    colored = plt.get_cmap("jet")(heat)[..., :3]
    return np.clip((1.0 - alpha) * frame_rgb + alpha * colored, 0.0, 1.0)


def load_background(background_dir: Path, pattern_id: str, image_size: int = 224) -> np.ndarray:
    path = background_dir / f"{pattern_id}.png"
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as file:
        return list(csv.DictReader(file))


def area_field(threshold: float) -> str:
    suffix = str(threshold).replace(".", "p")
    return f"active_area_fraction_t{suffix}"


def plot_entropy_axis(ax: plt.Axes, entropy_rows: list[dict[str, str]], pattern_order: list[str]) -> None:
    by_pattern = {str(row["background"]): row for row in entropy_rows}
    x = np.arange(len(pattern_order))
    entropies = [float(by_pattern[pattern]["pattern_entropy"]) for pattern in pattern_order]
    accuracies = [100.0 * float(by_pattern[pattern]["accuracy"]) for pattern in pattern_order]
    labels = [f"Pattern {pattern}" for pattern in pattern_order]

    bars = ax.bar(x, entropies, color="#7d91c2", edgecolor="#28324a", linewidth=0.8, label="Clean pattern entropy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Clean pattern entropy")
    ax.set_xlabel("Background pattern")
    ax.set_title("Static pattern entropy and accuracy, reordered")
    ax.grid(axis="y", alpha=0.35, linestyle=":")
    ax.set_ylim(0.0, max(entropies) * 1.18)

    ax2 = ax.twinx()
    ax2.plot(x, accuracies, color="#bd4e3d", marker="o", linewidth=2.0, label="Accuracy")
    ax2.set_ylabel("Prediction accuracy (%)")
    ax2.set_ylim(min(accuracies) - 1.0, max(accuracies) + 1.0)
    for xpos, accuracy in zip(x, accuracies):
        ax2.text(xpos, accuracy + 0.18, f"{accuracy:.1f}%", color="#8c3a31", ha="center", fontsize=8)

    handles = [bars, ax2.lines[0]]
    labels = ["Clean pattern entropy", "Accuracy"]
    ax.legend(handles, labels, loc="upper right", fontsize=8)


def build_area_table(
    area_rows: list[dict[str, str]],
    pattern_order: list[str],
    thresholds: list[float],
) -> tuple[list[str], list[list[str]]]:
    by_pattern: dict[str, list[dict[str, str]]] = {}
    for row in area_rows:
        by_pattern.setdefault(str(row["pattern_id"]), []).append(row)

    columns = ["Pattern", "n"] + [f"Area >= {threshold:.2f}" for threshold in thresholds]
    table_rows: list[list[str]] = []
    for pattern in pattern_order:
        rows = by_pattern[pattern]
        n = int(sum(int(row["sample_count"]) for row in rows))
        values = []
        for threshold in thresholds:
            mean_area = np.mean([100.0 * float(row[area_field(threshold)]) for row in rows])
            values.append(f"{mean_area:.1f}%")
        table_rows.append([f"P{pattern}", str(n), *values])
    return columns, table_rows


def plot_area_table(ax: plt.Axes, columns: list[str], rows: list[list[str]]) -> None:
    ax.axis("off")
    ax.set_title("Thresholded CAM active area, mean over C0-C9", loc="left", pad=12)
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.12, 1.0, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#334155")
        else:
            cell.set_facecolor("#f8fafc" if row_idx % 2 else "#e5e7eb")
    ax.text(
        0.0,
        0.0,
        "Area is measured on the 14x14 tubelet CAM grid after per-group min-max normalization.",
        fontsize=8,
        ha="left",
        va="bottom",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradcam-dir", type=Path, default=DEFAULT_GRADCAM_DIR)
    parser.add_argument("--entropy-csv", type=Path, default=DEFAULT_ENTROPY_CSV)
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BACKGROUND_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pattern-order", nargs="+", default=["1", "3", "4", "2"])
    parser.add_argument("--class-count", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aggregate_dir = args.gradcam_dir / "aggregate_maps"
    area_csv = args.gradcam_dir / "pattern_viscosity_gradcam_area_summary.csv"
    entropy_rows = read_rows(args.entropy_csv)
    area_rows = read_rows(area_csv)
    pattern_order = [str(pattern) for pattern in args.pattern_order]
    thresholds = [0.25, 0.50, 0.75]

    backgrounds = {pattern: load_background(args.background_dir, pattern) for pattern in pattern_order}
    area_by_key = {
        (str(row["pattern_id"]), int(row["true_viscosity_class"])): row
        for row in area_rows
    }

    fig = plt.figure(figsize=(22, 15), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=[4.5, 1.45], hspace=0.22)
    top = outer[0].subgridspec(len(pattern_order), args.class_count, hspace=0.38, wspace=0.04)
    axes = np.empty((len(pattern_order), args.class_count), dtype=object)

    for row_idx, pattern in enumerate(pattern_order):
        for true_class in range(args.class_count):
            ax = fig.add_subplot(top[row_idx, true_class])
            axes[row_idx, true_class] = ax
            map_path = aggregate_dir / f"pattern_{pattern}_class_{true_class}_mean_spatial_gradcam.npy"
            row = area_by_key.get((pattern, true_class))
            if row is None or not map_path.exists():
                ax.axis("off")
                continue
            mean_map = normalize(np.load(map_path))
            tile = overlay_frame(backgrounds[pattern], mean_map)
            ax.imshow(tile)
            ax.set_title(f"C{true_class}\nn={int(row['sample_count'])}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_idx, 0].set_ylabel(f"Pattern {pattern}", fontsize=11, rotation=90, labelpad=14)

    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.28)
    ax_entropy = fig.add_subplot(bottom[0, 0])
    plot_entropy_axis(ax_entropy, entropy_rows, pattern_order)
    ax_table = fig.add_subplot(bottom[0, 1])
    columns, table_rows = build_area_table(area_rows, pattern_order, thresholds)
    plot_area_table(ax_table, columns, table_rows)

    fig.suptitle(
        "No-RPM window30 Grad-CAM overlays by pattern and true viscosity class",
        fontsize=16,
        y=0.992,
    )
    fig.text(
        0.5,
        0.972,
        "Top: accumulated CAM overlays only. Bottom: reordered entropy/accuracy graph and thresholded active-area table.",
        ha="center",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
