#!/usr/bin/env python3
"""Summarize completed no-RPM policy runs against the transfer target."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "rebuild_reproduction"


RUNS = [
    {
        "name": "repro_realonly_993_batch8_normal_no_rpm_lr3e5_ep70",
        "role": "real-only baseline",
        "target": None,
    },
    {
        "name": "repro_realonly_993_batch8_normal_no_rpm_lr1e5_ep90",
        "role": "real-only recovery",
        "target": None,
    },
    {
        "name": "repro_transfer_993_batch8_normal_no_rpm_from_synth_no_rpm_lr3e5_ep70",
        "role": "synthetic-transfer target",
        "target": 0.9001,
    },
    {
        "name": "dual_pattern_transfer_no_rpm_from_synth_no_rpm_ep70",
        "role": "dual-pattern synthetic-transfer",
        "target": None,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT / "no_rpm_policy_report"),
        help="Directory for no-RPM policy summary artifacts.",
    )
    parser.add_argument(
        "--require-accepted",
        action="store_true",
        help="Exit nonzero unless all metrics exist and the transfer target is met.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metrics_path(run_name: str) -> Path:
    return OUTPUT_ROOT / run_name / "confusion_matrix" / f"{run_name}_metrics.json"


def reliability_path(run_name: str) -> Path:
    return OUTPUT_ROOT / run_name / "reliability_plots" / f"{run_name}_metrics.json"


def log_path(run_name: str) -> Path:
    return OUTPUT_ROOT / "logs" / f"{run_name}.log"


def wandb_from_log(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {"run_id": None, "url": None}
    text = path.read_text(errors="ignore")
    url_matches = re.findall(r"https://wandb\.ai/\S+/runs/([A-Za-z0-9_-]+)", text)
    url_match = re.search(r"https://wandb\.ai/\S+/runs/[A-Za-z0-9_-]+", text)
    local_matches = re.findall(r"wandb/run-\d{8}_\d{6}-([A-Za-z0-9_-]+)", text)
    run_id = url_matches[-1] if url_matches else (local_matches[-1] if local_matches else None)
    return {"run_id": run_id, "url": url_match.group(0) if url_match else None}


def run_record(spec: dict[str, Any]) -> dict[str, Any]:
    name = spec["name"]
    metrics = read_json(metrics_path(name))
    reliability = read_json(reliability_path(name))
    wandb = wandb_from_log(log_path(name))
    accuracy = None if metrics is None else float(metrics["accuracy"])
    ece = None if reliability is None else reliability.get("ece")
    target = spec["target"]
    return {
        "name": name,
        "role": spec["role"],
        "metrics_path": str(metrics_path(name)),
        "reliability_path": str(reliability_path(name)),
        "log_path": str(log_path(name)),
        "wandb_run_id": wandb["run_id"],
        "wandb_url": wandb["url"],
        "status": "complete" if metrics is not None else "missing_metrics",
        "accuracy": accuracy,
        "ece": ece,
        "target": target,
        "target_met": None if target is None or accuracy is None else accuracy >= target,
    }


def markdown(records: list[dict[str, Any]]) -> str:
    completed_realonly = [
        record for record in records if record["role"].startswith("real-only") and record["accuracy"] is not None
    ]
    completed_transfer = [
        record for record in records if record["role"] == "synthetic-transfer target" and record["accuracy"] is not None
    ]
    best_realonly = max(completed_realonly, key=lambda item: item["accuracy"], default=None)
    transfer = completed_transfer[0] if completed_transfer else None
    transfer_delta = None
    if best_realonly and transfer:
        transfer_delta = transfer["accuracy"] - best_realonly["accuracy"]

    lines = [
        "# No-RPM Policy Report",
        "",
        "Policy: no queued run should pass RPM conditioning to the model.",
        "",
        "| Run | Role | Status | Accuracy | ECE | Target | Target Met |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        accuracy = "" if record["accuracy"] is None else f"{record['accuracy']:.4f}"
        ece = "" if record["ece"] is None else f"{float(record['ece']):.4f}"
        target = "" if record["target"] is None else f"{record['target']:.4f}"
        target_met = "" if record["target_met"] is None else str(record["target_met"])
        lines.append(
            f"| `{record['name']}` | {record['role']} | {record['status']} | {accuracy} | {ece} | {target} | {target_met} |"
        )

    lines.extend(["", "## W&B"])
    for record in records:
        if record.get("wandb_run_id"):
            lines.append(f"- `{record['name']}`: `{record['wandb_run_id']}`")
        else:
            lines.append(f"- `{record['name']}`: missing from log")

    lines.extend(["", "## Acceptance"])
    if transfer is None:
        lines.append("- Transfer target is not yet judged because transfer metrics are missing.")
    elif transfer["target_met"]:
        lines.append(f"- Transfer target met: accuracy {transfer['accuracy']:.4f} >= 0.9001.")
    else:
        lines.append(f"- Transfer target not met: accuracy {transfer['accuracy']:.4f} < 0.9001.")

    if best_realonly is None:
        lines.append("- Best no-RPM real-only result is not yet known.")
    else:
        lines.append(f"- Best no-RPM real-only result: `{best_realonly['name']}` at {best_realonly['accuracy']:.4f}.")

    if transfer_delta is not None:
        sign = "+" if transfer_delta >= 0 else ""
        lines.append(f"- Transfer minus best real-only: {sign}{transfer_delta:.4f}.")

    return "\n".join(lines) + "\n"


def acceptance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    all_metrics_present = all(record["status"] == "complete" for record in records)
    transfer_target_met = next(
        (record["target_met"] for record in records if record["role"] == "synthetic-transfer target"),
        None,
    )
    accepted = bool(all_metrics_present and transfer_target_met)
    return {
        "all_metrics_present": all_metrics_present,
        "transfer_target_met": transfer_target_met,
        "accepted": accepted,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [run_record(spec) for spec in RUNS]
    summary = {
        "records": records,
        **acceptance_summary(records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "summary.md").write_text(markdown(records))
    print(json.dumps(summary, indent=2))
    return 1 if args.require_accepted and not summary["accepted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
