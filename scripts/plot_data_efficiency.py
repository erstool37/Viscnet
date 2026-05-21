#!/usr/bin/env python3
"""Write data-efficiency tables and plots from checker metrics."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "rebuild_reproduction"
METRICS_TABLE = OUTPUT_ROOT / "metrics_table.json"
REFERENCE_PATH = ROOT / "configs" / "rebuild" / "reference_metrics.json"
CSV_PATH = OUTPUT_ROOT / "data_efficiency_metrics.csv"
JSON_PATH = OUTPUT_ROOT / "data_efficiency_metrics.json"
PLOT_PATH = OUTPUT_ROOT / "data_efficiency_curve.png"
SIZES = [300, 400, 500, 600, 700, 800, 900, 993]


def sample_size(config: str, prefix: str) -> int | None:
    match = re.search(rf"/{prefix}_(\d+)\.yaml$", config)
    return int(match.group(1)) if match else None


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = json.loads(METRICS_TABLE.read_text()) if METRICS_TABLE.exists() else []
    refs = json.loads(REFERENCE_PATH.read_text())

    by_size = {
        size: {
            "samples": size,
            "realonly_accuracy": None,
            "realonly_status": "pending",
            "realonly_target": refs["real_only_curve"][str(size)]["accuracy"],
            "transfer_accuracy": None,
            "transfer_status": "pending",
            "transfer_target": refs["transfer_curve"][str(size)]["accuracy"],
            "transfer_gain": None,
        }
        for size in SIZES
    }

    for row in rows:
        config = row.get("config", "")
        real_size = sample_size(config, "realonly")
        transfer_size = sample_size(config, "transfer")
        if real_size in by_size:
            by_size[real_size]["realonly_accuracy"] = row.get("observed_accuracy")
            by_size[real_size]["realonly_status"] = row.get("status")
        if transfer_size in by_size:
            by_size[transfer_size]["transfer_accuracy"] = row.get("observed_accuracy")
            by_size[transfer_size]["transfer_status"] = row.get("status")

    records = []
    for size in SIZES:
        record = by_size[size]
        real = record["realonly_accuracy"]
        transfer = record["transfer_accuracy"]
        if real is not None and transfer is not None:
            record["transfer_gain"] = transfer - real
        records.append(record)

    JSON_PATH.write_text(json.dumps(records, indent=2) + "\n")
    with CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    sizes = [record["samples"] for record in records]
    real_targets = [record["realonly_target"] for record in records]
    transfer_targets = [record["transfer_target"] for record in records]
    real_observed = [record["realonly_accuracy"] for record in records]
    transfer_observed = [record["transfer_accuracy"] for record in records]

    ax.plot(
        sizes, real_targets, color="#6b7280", linestyle="--", marker="o", linewidth=1.5, label="Legacy real-only target"
    )
    ax.plot(
        sizes,
        transfer_targets,
        color="#9ca3af",
        linestyle="--",
        marker="o",
        linewidth=1.5,
        label="Legacy transfer target",
    )
    if any(value is not None for value in real_observed):
        ax.plot(sizes, real_observed, color="#2563eb", marker="o", linewidth=2.0, label="30ep microbatch real-only")
    if any(value is not None for value in transfer_observed):
        ax.plot(sizes, transfer_observed, color="#dc2626", marker="o", linewidth=2.0, label="30ep microbatch transfer")

    ax.set_title("Viscnet Data Efficiency: 30-Epoch Microbatch")
    ax.set_xlabel("Real training samples")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    print(f"Wrote {CSV_PATH.relative_to(ROOT)}")
    print(f"Wrote {JSON_PATH.relative_to(ROOT)}")
    print(f"Wrote {PLOT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
