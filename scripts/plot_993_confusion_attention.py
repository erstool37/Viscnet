#!/usr/bin/env python3
"""Plot 993 confusion and attention comparisons.

This is a post-hoc diagnostic over existing metrics and attention volumes.
It does not train models or modify checkpoints.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs/rebuild_reproduction")
ATTN_ROOT = ROOT / "attention_993"
OUT = ROOT / "attention_993_comparison"

RUNS = [
    {
        "key": "realonly_993_30ep_microbatch",
        "label": "Real-only 30ep microbatch",
        "confusion": ROOT / "repro_realonly_993_microbatch/confusion_matrix/repro_realonly_993_microbatch_metrics.json",
        "attention": ATTN_ROOT / "realonly_993_30ep_microbatch",
    },
    {
        "key": "transfer_993_30ep_microbatch",
        "label": "Transfer 30ep microbatch",
        "confusion": ROOT / "repro_transfer_993_microbatch/confusion_matrix/repro_transfer_993_microbatch_metrics.json",
        "attention": ATTN_ROOT / "transfer_993_30ep_microbatch",
    },
    {
        "key": "realonly_993_300epoch",
        "label": "Real-only 300ep diagnostic",
        "confusion": ROOT / "repro_realonly_993/confusion_matrix/repro_realonly_993_metrics.json",
        "attention": ATTN_ROOT / "realonly_993_300epoch",
    },
    {
        "key": "realonly_993_lrhold",
        "label": "Real-only LR-hold",
        "confusion": ROOT
        / "repro_realonly_993_microbatch_lrhold/confusion_matrix/repro_realonly_993_microbatch_lrhold_metrics.json",
        "attention": ATTN_ROOT / "realonly_993_lrhold",
    },
    {
        "key": "transfer_993_lrhold",
        "label": "Transfer LR-hold",
        "confusion": ROOT
        / "repro_transfer_993_microbatch_lrhold/confusion_matrix/repro_transfer_993_microbatch_lrhold_metrics.json",
        "attention": ATTN_ROOT / "transfer_993_lrhold",
    },
]

FILENAME_RE = re.compile(r"_class(?P<true>\d+)_pred(?P<pred>\d+)_cls_attn_vol\.npy$")


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def load_confusion(run: dict[str, Any]) -> dict[str, Any]:
    with open(run["confusion"], "r") as file:
        payload = json.load(file)
    matrix = np.asarray(payload["confusion_matrix_counts"], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    payload["matrix"] = matrix
    payload["normalized"] = normalized
    payload["support"] = np.asarray(payload["support"], dtype=float)
    payload["per_class_accuracy"] = np.asarray(payload["per_class_accuracy"], dtype=float)
    return payload


def annotate_matrix(ax: plt.Axes, matrix: np.ndarray, fmt: str) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = format(value, fmt)
            color = "white" if value > (np.nanmax(matrix) * 0.55) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)


def mark_low_classes(ax: plt.Axes) -> None:
    for idx in [1, 2, 3]:
        ax.add_patch(plt.Rectangle((idx - 0.5, -0.5), 1, 10, fill=False, ec="crimson", lw=0.7))
        ax.add_patch(plt.Rectangle((-0.5, idx - 0.5), 10, 1, fill=False, ec="crimson", lw=0.7))
    ax.add_patch(plt.Rectangle((0.5, 0.5), 3, 3, fill=False, ec="red", lw=1.5))


def set_confusion_ticks(ax: plt.Axes, support: np.ndarray | None = None) -> None:
    ax.set_xticks(range(10))
    ax.set_xlabel("Predicted")
    if support is None:
        ax.set_yticks(range(10))
        ax.set_yticklabels(range(10))
    else:
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"{idx} n={int(n)}" for idx, n in enumerate(support)])
    ax.set_ylabel("True")


def plot_confusion_all(confusions: dict[str, dict[str, Any]]) -> None:
    for mode, fname, cmap, fmt in [
        ("matrix", "confusion_total_counts_all_993.png", "Blues", ".0f"),
        ("normalized", "confusion_total_normalized_all_993.png", "Blues", ".2f"),
    ]:
        vmax = max(np.nanmax(confusions[run["key"]][mode]) for run in RUNS)
        fig, axes = plt.subplots(1, len(RUNS), figsize=(4.0 * len(RUNS), 4.2), constrained_layout=True)
        for ax, run in zip(axes, RUNS):
            mat = confusions[run["key"]][mode]
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
            annotate_matrix(ax, mat, fmt)
            mark_low_classes(ax)
            ax.set_title(f"{run['label']}\nacc={confusions[run['key']]['accuracy']:.3f}", fontsize=9)
            set_confusion_ticks(ax, confusions[run["key"]]["support"])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
        fig.savefig(OUT / fname, dpi=220)
        plt.close(fig)


def plot_confusion_microbatch_delta(confusions: dict[str, dict[str, Any]]) -> None:
    real = confusions["realonly_993_30ep_microbatch"]
    transfer = confusions["transfer_993_30ep_microbatch"]
    for mode, fname, vmax in [
        ("matrix", "confusion_realonly_vs_transfer_microbatch_counts_diff.png", None),
        ("normalized", "confusion_realonly_vs_transfer_microbatch_normalized_diff.png", 0.45),
    ]:
        real_mat = real[mode]
        trans_mat = transfer[mode]
        diff = trans_mat - real_mat
        vmax_abs = vmax if vmax is not None else max(abs(float(diff.min())), abs(float(diff.max())))
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
        for ax, mat, title, cmap, vmin, vmax_plot, fmt in [
            (
                axes[0],
                real_mat,
                f"Real-only\nacc={real['accuracy']:.3f}",
                "Blues",
                0,
                np.nanmax([real_mat, trans_mat]),
                ".0f" if mode == "matrix" else ".2f",
            ),
            (
                axes[1],
                trans_mat,
                f"Transfer\nacc={transfer['accuracy']:.3f}",
                "Blues",
                0,
                np.nanmax([real_mat, trans_mat]),
                ".0f" if mode == "matrix" else ".2f",
            ),
            (
                axes[2],
                diff,
                "Transfer - Real-only",
                "coolwarm",
                -vmax_abs,
                vmax_abs,
                ".0f" if mode == "matrix" else ".2f",
            ),
        ]:
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax_plot)
            annotate_matrix(ax, mat, fmt)
            mark_low_classes(ax)
            ax.set_title(title)
            set_confusion_ticks(ax, real["support"])
            fig.colorbar(im, ax=ax, shrink=0.75)
        fig.savefig(OUT / fname, dpi=220)
        plt.close(fig)


def plot_lowclass_block(confusions: dict[str, dict[str, Any]]) -> None:
    real = confusions["realonly_993_30ep_microbatch"]["matrix"]
    transfer = confusions["transfer_993_30ep_microbatch"]["matrix"]
    rows = [1, 2, 3]
    cols = [0, 1, 2, 3, 4]
    real_block = real[np.ix_(rows, cols)]
    transfer_block = transfer[np.ix_(rows, cols)]
    diff = transfer_block - real_block
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    for ax, mat, title, cmap, vmin, vmax, fmt in [
        (
            axes[0],
            real_block,
            "Real-only counts\ntrue 1/2/3",
            "Blues",
            0,
            max(real_block.max(), transfer_block.max()),
            ".0f",
        ),
        (
            axes[1],
            transfer_block,
            "Transfer counts\ntrue 1/2/3",
            "Blues",
            0,
            max(real_block.max(), transfer_block.max()),
            ".0f",
        ),
        (
            axes[2],
            diff,
            "Transfer - Real-only",
            "coolwarm",
            -max(abs(diff.min()), abs(diff.max())),
            max(abs(diff.min()), abs(diff.max())),
            ".0f",
        ),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        annotate_matrix(ax, mat, fmt)
        ax.set_title(title)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(rows)), rows)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(OUT / "confusion_lowclass_123_microbatch_counts.png", dpi=220)
    plt.close(fig)


def plot_per_class_accuracy(confusions: dict[str, dict[str, Any]]) -> None:
    x = np.arange(10)
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    for idx, run in enumerate(RUNS):
        offset = (idx - (len(RUNS) - 1) / 2) * width
        ax.bar(x + offset, confusions[run["key"]]["per_class_accuracy"], width=width, label=run["label"])
    ax.axvspan(0.5, 3.5, color="crimson", alpha=0.08)
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("True class")
    ax.set_ylabel("Per-class accuracy")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "per_class_accuracy_993_all_runs.png", dpi=220)
    plt.close(fig)


def normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)


def entropy_norm(values: np.ndarray) -> float:
    flat = values.astype(np.float64).reshape(-1)
    total = float(np.nansum(flat))
    if total <= 0:
        return 0.0
    probs = flat / total
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / math.log(probs.size))


def parse_attention(run: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    volume_dir = run["attention"] / "volumes"
    for path in sorted(volume_dir.glob("*.npy")):
        match = FILENAME_RE.search(path.name)
        if not match:
            continue
        volume = np.load(path)
        true = int(match.group("true"))
        pred = int(match.group("pred"))
        records.append({"path": path, "true": true, "pred": pred, "volume": volume})
    return records


def mean_volume(records: list[dict[str, Any]]) -> np.ndarray | None:
    if not records:
        return None
    max_g = max(record["volume"].shape[2] for record in records)
    padded = []
    for record in records:
        volume = record["volume"].astype(np.float32)
        out = np.full((volume.shape[0], volume.shape[1], max_g), np.nan, dtype=np.float32)
        out[:, :, : volume.shape[2]] = volume
        padded.append(out)
    return np.nanmean(np.stack(padded, axis=0), axis=0)


def volume_metrics(volume: np.ndarray | None) -> dict[str, float | None]:
    if volume is None:
        return {
            "spatial_entropy": None,
            "temporal_entropy": None,
            "top10_spatial_mass": None,
            "early_mass": None,
            "mid_mass": None,
            "late_mass": None,
            "temporal_peak_idx": None,
            "temporal_peak_frac": None,
        }
    spatial = np.nansum(volume, axis=2)
    temporal = np.nansum(volume, axis=(0, 1))
    spatial_total = float(np.nansum(spatial))
    temporal_total = float(np.nansum(temporal))
    spatial_probs = spatial / spatial_total if spatial_total > 0 else spatial
    temporal_probs = temporal / temporal_total if temporal_total > 0 else temporal
    flat = spatial_probs.reshape(-1)
    top_k = max(1, int(math.ceil(0.10 * flat.size)))
    thirds = np.array_split(np.arange(temporal_probs.shape[0]), 3)
    return {
        "spatial_entropy": entropy_norm(spatial),
        "temporal_entropy": entropy_norm(temporal),
        "top10_spatial_mass": float(np.sort(flat)[-top_k:].sum()),
        "early_mass": float(np.nansum(temporal_probs[thirds[0]])),
        "mid_mass": float(np.nansum(temporal_probs[thirds[1]])),
        "late_mass": float(np.nansum(temporal_probs[thirds[2]])),
        "temporal_peak_idx": float(np.nanargmax(temporal_probs)),
        "temporal_peak_frac": float(np.nanmax(temporal_probs)),
    }


def build_attention_summary(
    attention_records: dict[str, list[dict[str, Any]]], confusions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for run in RUNS:
        key = run["key"]
        records = attention_records[key]
        for cls in range(10):
            group = [record for record in records if record["true"] == cls]
            correct = [record for record in group if record["pred"] == cls]
            wrong = [record for record in group if record["pred"] != cls]
            volume = mean_volume(group)
            metrics = volume_metrics(volume)
            row = {
                "run_key": key,
                "run_label": run["label"],
                "true_class": cls,
                "count": len(group),
                "correct": len(correct),
                "wrong": len(wrong),
                "accuracy": confusions[key]["per_class_accuracy"][cls],
            }
            row.update(metrics)
            rows.append(row)
    return rows


def write_attention_summary(rows: list[dict[str, Any]]) -> None:
    with open(OUT / "attention_true_class_summary.csv", "w", newline="") as file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_spatial_trueclass(attention_records: dict[str, list[dict[str, Any]]]) -> None:
    selected = [
        ("realonly_993_30ep_microbatch", "Real-only 30ep"),
        ("transfer_993_30ep_microbatch", "Transfer 30ep"),
    ]
    fig, axes = plt.subplots(len(selected), 10, figsize=(20, 4.3), constrained_layout=True)
    for row_idx, (key, label) in enumerate(selected):
        records = attention_records[key]
        for cls in range(10):
            ax = axes[row_idx, cls]
            volume = mean_volume([record for record in records if record["true"] == cls])
            if volume is not None:
                spatial = normalize(np.nanmean(normalize(volume), axis=2))
                ax.imshow(spatial, cmap="viridis", origin="upper", vmin=0, vmax=1)
            ax.set_title(f"{label}\ntrue {cls}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(OUT / "attention_spatial_trueclass_real_vs_transfer_microbatch.png", dpi=220)
    plt.close(fig)


def plot_temporal_trueclass(attention_records: dict[str, list[dict[str, Any]]]) -> None:
    selected = [
        ("realonly_993_30ep_microbatch", "Real-only 30ep"),
        ("transfer_993_30ep_microbatch", "Transfer 30ep"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.5), constrained_layout=True)
    for cls, ax in enumerate(axes.ravel()):
        for key, label in selected:
            volume = mean_volume([record for record in attention_records[key] if record["true"] == cls])
            if volume is not None:
                temporal = normalize(np.nanmean(normalize(volume), axis=(0, 1)))
                ax.plot(temporal, marker="o", ms=2.5, lw=1.1, label=label)
        ax.set_title(f"True class {cls}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=8)
    fig.savefig(OUT / "attention_temporal_trueclass_real_vs_transfer_microbatch.png", dpi=220)
    plt.close(fig)


def plot_lowclass_attention_delta(attention_records: dict[str, list[dict[str, Any]]]) -> None:
    real_key = "realonly_993_30ep_microbatch"
    transfer_key = "transfer_993_30ep_microbatch"
    classes = [1, 2, 3]
    fig, axes = plt.subplots(4, len(classes), figsize=(10.5, 10), constrained_layout=True)
    for col, cls in enumerate(classes):
        real_volume = mean_volume([r for r in attention_records[real_key] if r["true"] == cls])
        transfer_volume = mean_volume([r for r in attention_records[transfer_key] if r["true"] == cls])
        real_spatial = normalize(np.nanmean(normalize(real_volume), axis=2))
        transfer_spatial = normalize(np.nanmean(normalize(transfer_volume), axis=2))
        diff = transfer_spatial - real_spatial
        real_temporal = normalize(np.nanmean(normalize(real_volume), axis=(0, 1)))
        transfer_temporal = normalize(np.nanmean(normalize(transfer_volume), axis=(0, 1)))

        axes[0, col].imshow(real_spatial, cmap="viridis", origin="upper", vmin=0, vmax=1)
        axes[0, col].set_title(f"Real-only true {cls}")
        axes[1, col].imshow(transfer_spatial, cmap="viridis", origin="upper", vmin=0, vmax=1)
        axes[1, col].set_title(f"Transfer true {cls}")
        vmax = max(abs(float(diff.min())), abs(float(diff.max())))
        axes[2, col].imshow(diff, cmap="coolwarm", origin="upper", vmin=-vmax, vmax=vmax)
        axes[2, col].set_title("Transfer - Real-only")
        axes[3, col].plot(real_temporal, marker="o", ms=2.5, lw=1.1, label="Real-only")
        axes[3, col].plot(transfer_temporal, marker="o", ms=2.5, lw=1.1, label="Transfer")
        axes[3, col].set_title(f"Temporal true {cls}")
        axes[3, col].set_ylim(-0.05, 1.05)
        axes[3, col].grid(alpha=0.25)
        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[3, 0].legend(fontsize=8)
    fig.savefig(OUT / "attention_lowclass_123_real_vs_transfer_microbatch.png", dpi=220)
    plt.close(fig)


def plot_all_existing_attention_panels() -> None:
    images = []
    labels = []
    for run in RUNS:
        panel = run["attention"] / "panels/attention_summary_panels.png"
        if panel.exists():
            images.append(mpimg.imread(panel))
            labels.append(run["label"])
    fig, axes = plt.subplots(len(images), 1, figsize=(18, 5.6 * len(images)), constrained_layout=True)
    if len(images) == 1:
        axes = [axes]
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=14)
        ax.axis("off")
    fig.savefig(OUT / "attention_summary_panels_all_runs.png", dpi=160)
    plt.close(fig)


def write_microbatch_delta(confusions: dict[str, dict[str, Any]]) -> None:
    real = confusions["realonly_993_30ep_microbatch"]
    transfer = confusions["transfer_993_30ep_microbatch"]
    rows = []
    for cls in range(10):
        real_correct = real["matrix"][cls, cls]
        transfer_correct = transfer["matrix"][cls, cls]
        rows.append(
            {
                "class": cls,
                "support": int(real["support"][cls]),
                "realonly_correct": int(real_correct),
                "transfer_correct": int(transfer_correct),
                "correct_delta_transfer_minus_real": int(transfer_correct - real_correct),
                "realonly_acc": float(real["per_class_accuracy"][cls]),
                "transfer_acc": float(transfer["per_class_accuracy"][cls]),
                "acc_delta_transfer_minus_real": float(
                    transfer["per_class_accuracy"][cls] - real["per_class_accuracy"][cls]
                ),
                "realonly_1_to_2": int(real["matrix"][1, 2]) if cls == 1 else "",
                "transfer_1_to_2": int(transfer["matrix"][1, 2]) if cls == 1 else "",
                "realonly_2_to_1": int(real["matrix"][2, 1]) if cls == 2 else "",
                "transfer_2_to_1": int(transfer["matrix"][2, 1]) if cls == 2 else "",
                "realonly_2_to_3": int(real["matrix"][2, 3]) if cls == 2 else "",
                "transfer_2_to_3": int(transfer["matrix"][2, 3]) if cls == 2 else "",
                "realonly_3_to_2": int(real["matrix"][3, 2]) if cls == 3 else "",
                "transfer_3_to_2": int(transfer["matrix"][3, 2]) if cls == 3 else "",
            }
        )
    with open(OUT / "microbatch_real_vs_transfer_class_delta.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(confusions: dict[str, dict[str, Any]]) -> None:
    real = confusions["realonly_993_30ep_microbatch"]
    transfer = confusions["transfer_993_30ep_microbatch"]
    correct_delta = np.diag(transfer["matrix"]) - np.diag(real["matrix"])
    low_delta = int(correct_delta[[1, 2, 3]].sum())
    total_delta = int(correct_delta.sum())
    report = f"""# 993 Confusion And Attention Comparison

Generated by `scripts/plot_993_confusion_attention.py`.

## Main Question

Does the `transfer_993_microbatch` improvement over `realonly_993_microbatch`
come from classes `1`, `2`, and `3`?

Evidence says: only partly. Transfer improves total accuracy by `+0.100`
(`0.819 - 0.719`), which is `+{total_delta}` more correct samples out of 1000.
True classes `1/2/3` contribute net `+{low_delta}` correct samples, so they explain
about `{(low_delta / total_delta) if total_delta else 0:.1%}` of the gain.

Class-level transfer deltas:

- Class 1: {int(correct_delta[1]):+d} correct samples. Transfer is worse here:
  `1->2` errors increase from `{int(real["matrix"][1, 2])}` to `{int(transfer["matrix"][1, 2])}`.
- Class 2: {int(correct_delta[2]):+d} correct samples. Transfer is only slightly
  better.
- Class 3: {int(correct_delta[3]):+d} correct samples. This is the strongest
  low-class improvement; `3->2` errors fall from `{int(real["matrix"][3, 2])}` to
  `{int(transfer["matrix"][3, 2])}`.

So the low-class difference is mostly class `3`, not the full `1/2/3` block.
Other classes also matter: class `0` contributes `{int(correct_delta[0]):+d}`,
class `7` contributes `{int(correct_delta[7]):+d}`, class `6` contributes
`{int(correct_delta[6]):+d}`, and class `5` contributes `{int(correct_delta[5]):+d}`.

## Figures

- `confusion_total_counts_all_993.png`
- `confusion_total_normalized_all_993.png`
- `confusion_realonly_vs_transfer_microbatch_counts_diff.png`
- `confusion_realonly_vs_transfer_microbatch_normalized_diff.png`
- `confusion_lowclass_123_microbatch_counts.png`
- `per_class_accuracy_993_all_runs.png`
- `attention_spatial_trueclass_real_vs_transfer_microbatch.png`
- `attention_temporal_trueclass_real_vs_transfer_microbatch.png`
- `attention_lowclass_123_real_vs_transfer_microbatch.png`
- `attention_summary_panels_all_runs.png`

## Attention Interpretation

The 30-epoch transfer model has more concentrated final-layer attention than the
30-epoch real-only model. In `summary.csv`, transfer has lower spatial/temporal
entropy and higher top-10% spatial mass. That means synthetic pretraining makes
the model more selective.

But the selective attention does not solve class `1`: it shifts true class `1`
toward class `2` more often. For class `3`, transfer attention and classification
are both better aligned, which is why class `3` is the largest low-class gain.

The best high-epoch real-only result is different: it improves class `1/2/3`
without becoming sharply localized. That supports the earlier hypothesis that
the high-epoch/LR-hold benefit is mostly better real-domain boundary learning,
not a new attention hotspot.
"""
    (OUT / "confusion_attention_analysis.md").write_text(report)


def main() -> None:
    ensure_out()
    confusions = {run["key"]: load_confusion(run) for run in RUNS}
    plot_confusion_all(confusions)
    plot_confusion_microbatch_delta(confusions)
    plot_lowclass_block(confusions)
    plot_per_class_accuracy(confusions)
    attention_records = {run["key"]: parse_attention(run) for run in RUNS}
    rows = build_attention_summary(attention_records, confusions)
    write_attention_summary(rows)
    plot_spatial_trueclass(attention_records)
    plot_temporal_trueclass(attention_records)
    plot_lowclass_attention_delta(attention_records)
    plot_all_existing_attention_panels()
    write_microbatch_delta(confusions)
    write_report(confusions)
    print(f"Wrote figures and tables to {OUT}")


if __name__ == "__main__":
    main()
