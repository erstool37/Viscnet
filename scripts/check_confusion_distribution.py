#!/usr/bin/env python3
"""Summarize whether frozen real-video predictions are class-distributed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _string_keyed(labels, values, cast=float):
    return {str(label): cast(value) for label, value in zip(labels, values)}


def summarize_distribution(
    metrics,
    min_used_classes=8,
    max_predicted_class_share=0.35,
    max_zero_predicted_classes=2,
):
    counts = np.asarray(metrics["confusion_matrix_counts"], dtype=float)
    if counts.ndim != 2:
        raise ValueError("confusion_matrix_counts must be a 2D matrix")

    labels = metrics.get("labels", list(range(counts.shape[1])))
    if len(labels) != counts.shape[1]:
        raise ValueError("labels length must match confusion matrix columns")

    predicted_counts = counts.sum(axis=0)
    total = float(predicted_counts.sum())
    if total <= 0:
        predicted_shares = np.zeros_like(predicted_counts, dtype=float)
        max_share = 0.0
    else:
        predicted_shares = predicted_counts / total
        max_share = float(predicted_shares.max())

    zero_labels = [label for label, value in zip(labels, predicted_counts) if value == 0]
    used_classes = int(np.count_nonzero(predicted_counts))
    support = metrics.get("support", counts.sum(axis=1).astype(int).tolist())
    per_class_accuracy = metrics.get("per_class_accuracy")
    if per_class_accuracy is None:
        row_totals = counts.sum(axis=1)
        per_class_accuracy = np.divide(
            np.diag(counts),
            row_totals,
            out=np.zeros_like(row_totals, dtype=float),
            where=row_totals != 0,
        ).tolist()

    well_distributed = (
        used_classes >= int(min_used_classes)
        and max_share <= float(max_predicted_class_share)
        and len(zero_labels) <= int(max_zero_predicted_classes)
    )

    return {
        "accuracy": metrics.get("accuracy"),
        "predicted_class_counts": _string_keyed(labels, predicted_counts.astype(int).tolist(), int),
        "predicted_class_shares": _string_keyed(labels, predicted_shares.tolist(), float),
        "predicted_classes_used": used_classes,
        "max_predicted_class_share": max_share,
        "zero_predicted_classes": zero_labels,
        "support": _string_keyed(labels, support, int),
        "per_class_accuracy": _string_keyed(labels, per_class_accuracy, float),
        "well_distributed": well_distributed,
        "criteria": {
            "min_used_classes": int(min_used_classes),
            "max_predicted_class_share": float(max_predicted_class_share),
            "max_zero_predicted_classes": int(max_zero_predicted_classes),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_path", nargs="?", help="Confusion metrics JSON path")
    parser.add_argument("--metrics", dest="metrics_option", help="Confusion metrics JSON path")
    parser.add_argument("--output", help="Optional JSON summary output path")
    parser.add_argument("--min-used-classes", type=int, default=8)
    parser.add_argument("--max-predicted-class-share", type=float, default=0.35)
    parser.add_argument("--max-zero-predicted-classes", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = args.metrics_option or args.metrics_path
    if not metrics_path:
        raise SystemExit("metrics path is required")

    with Path(metrics_path).open("r") as file:
        metrics = json.load(file)
    summary = summarize_distribution(
        metrics,
        min_used_classes=args.min_used_classes,
        max_predicted_class_share=args.max_predicted_class_share,
        max_zero_predicted_classes=args.max_zero_predicted_classes,
    )

    payload = json.dumps(summary, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload)
    print(payload, end="")
    return 0 if summary["well_distributed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
