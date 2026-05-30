#!/usr/bin/env python3
"""Summarize raw-space and log-space regression errors from an error CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.csv)
    required = {"target", "pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{args.csv} is missing required columns: {sorted(missing)}")

    target = df["target"].to_numpy(dtype=float)
    pred = df["pred"].to_numpy(dtype=float)
    finite = np.isfinite(target) & np.isfinite(pred) & (target > 0.0) & (pred > 0.0)
    target = target[finite]
    pred = pred[finite]
    abs_err = np.abs(pred - target)
    ape = abs_err / target * 100.0
    summary = {
        "source_csv": str(args.csv),
        "n": int(target.size),
        "naive_mae": float(abs_err.mean()) if target.size else None,
        "log10_mae": float(np.abs(np.log10(pred) - np.log10(target)).mean()) if target.size else None,
        "ln_mae": float(np.abs(np.log(pred) - np.log(target)).mean()) if target.size else None,
        "mape_percent": float(ape.mean()) if target.size else None,
        "median_abs_error": float(np.median(abs_err)) if target.size else None,
        "median_ape_percent": float(np.median(ape)) if target.size else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        lines = [
            "# Regression Error Summary",
            "",
            f"- Source CSV: `{args.csv}`",
            f"- N: {summary['n']}",
            f"- Naive MAE: `{summary['naive_mae']}`",
            f"- Log10 MAE: `{summary['log10_mae']}`",
            f"- MAPE: `{summary['mape_percent']}%`",
            f"- Median absolute error: `{summary['median_abs_error']}`",
            f"- Median APE: `{summary['median_ape_percent']}%`",
        ]
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
