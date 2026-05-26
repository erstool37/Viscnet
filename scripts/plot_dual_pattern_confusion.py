#!/usr/bin/env python3
"""Draw paper-style count and normalized confusion matrices for dual-pattern runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Dual-pattern confusion matrix")
    return parser.parse_args()


def draw_matrix(matrix: np.ndarray, labels: list[int], title: str, output_path: Path, *, fmt: str, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = 1.0 if fmt == ".2f" else max(1.0, float(matrix.max()))
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Predicted viscosity class")
    ax.set_ylabel("True viscosity class")
    ax.set_xticks(range(len(labels)), [str(label) for label in labels])
    ax.set_yticks(range(len(labels)), [str(label) for label in labels])

    threshold = vmax * 0.55
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value == 0:
                continue
            color = "white" if value >= threshold else "black"
            text = f"{value:{fmt}}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.metrics)
    output_dir = Path(args.output_dir)
    metrics = json.loads(metrics_path.read_text())

    labels = [int(label) for label in metrics["labels"]]
    counts = np.array(metrics["confusion_matrix_counts"], dtype=float)
    normalized = np.array(metrics["confusion_matrix_normalized"], dtype=float)
    active = [idx for idx, support in enumerate(metrics["support"]) if support > 0]
    if active:
        active_labels = [labels[idx] for idx in active]
        active_counts = counts[np.ix_(active, active)]
        active_normalized = normalized[np.ix_(active, active)]
    else:
        active_labels = labels
        active_counts = counts
        active_normalized = normalized

    draw_matrix(
        counts,
        labels,
        f"{args.title} counts",
        output_dir / "dual_pattern_confusion_counts_full.png",
        fmt=".0f",
        cmap="Blues",
    )
    draw_matrix(
        normalized,
        labels,
        f"{args.title} normalized",
        output_dir / "dual_pattern_confusion_normalized_full.png",
        fmt=".2f",
        cmap="Blues",
    )
    draw_matrix(
        active_counts,
        active_labels,
        f"{args.title} counts, active classes",
        output_dir / "dual_pattern_confusion_counts_active.png",
        fmt=".0f",
        cmap="Blues",
    )
    draw_matrix(
        active_normalized,
        active_labels,
        f"{args.title} normalized, active classes",
        output_dir / "dual_pattern_confusion_normalized_active.png",
        fmt=".2f",
        cmap="Blues",
    )

    summary = {
        "metrics": str(metrics_path),
        "accuracy": metrics.get("accuracy"),
        "labels": labels,
        "active_labels": active_labels,
        "figures": sorted(path.name for path in output_dir.glob("dual_pattern_confusion_*.png")),
    }
    (output_dir / "dual_pattern_confusion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
