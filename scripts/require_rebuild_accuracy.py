#!/usr/bin/env python3
"""Fail unless a completed rebuild run meets an accuracy threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "rebuild_reproduction"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--min-accuracy", type=float, required=True)
    args = parser.parse_args()

    metrics_path = OUTPUT_ROOT / args.run_name / "confusion_matrix" / f"{args.run_name}_metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    accuracy = float(metrics["accuracy"])
    print(f"{args.run_name}: accuracy={accuracy:.4f}, required={args.min_accuracy:.4f}")
    if accuracy < args.min_accuracy:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
