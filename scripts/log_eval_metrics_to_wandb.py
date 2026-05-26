#!/usr/bin/env python3
"""Log completed evaluation metrics and artifacts to W&B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wandb


def flatten_numeric(prefix: str, payload: dict) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values[name] = value
        elif isinstance(value, dict):
            values.update(flatten_numeric(f"{name}/", value))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--predictions", default="")
    parser.add_argument("--project", default="re-rebuild-viscnet")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "jongwonsohn-seoul-national-university"))
    parser.add_argument("--job-type", default="evaluation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is not set; refusing to claim W&B provenance for this eval.")

    metrics_path = Path(args.metrics)
    output_dir = Path(args.output_dir)
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    config_payload = {
        "source_config": args.config,
        "metrics_path": str(metrics_path),
        "output_dir": str(output_dir),
        "prediction_mode": metrics.get("prediction_mode"),
        "checkpoint": metrics.get("checkpoint"),
        "test_root": metrics.get("test_root"),
        "num_windows_per_video": metrics.get("num_windows_per_video"),
        "evaluated_sample_count": metrics.get("evaluated_sample_count"),
    }
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        job_type=args.job_type,
        config=config_payload,
        reinit=True,
        resume="never",
    )
    wandb.log(flatten_numeric("", metrics))
    artifact = wandb.Artifact(args.run_name, type="eval-results")
    artifact.add_file(str(metrics_path))
    for candidate in [
        args.predictions,
        output_dir / "window21_clip_test_inference.png",
        output_dir / "window21_clip_test_inference_report.md",
    ]:
        if candidate and Path(candidate).exists():
            artifact.add_file(str(candidate))
    run.log_artifact(artifact)
    run.finish()
    print(json.dumps({"wandb_run_id": run.id, "wandb_url": run.url}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
