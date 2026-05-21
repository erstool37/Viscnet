#!/usr/bin/env python3
"""Run post-training rebuild checks and optional completion notification."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "rebuild_reproduction"
NOTIFY_ROOT = OUTPUT_ROOT / "notifications"
METRICS_TABLE = OUTPUT_ROOT / "metrics_table.json"
CHECKER_REPORT = OUTPUT_ROOT / "checker_report.md"
ANALYZER_REPORT = OUTPUT_ROOT / "analyzer_report.md"
DATA_EFFICIENCY_CSV = OUTPUT_ROOT / "data_efficiency_metrics.csv"
DATA_EFFICIENCY_JSON = OUTPUT_ROOT / "data_efficiency_metrics.json"
DATA_EFFICIENCY_PLOT = OUTPUT_ROOT / "data_efficiency_curve.png"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    value = value.strip() or "rebuild_training"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def run_command(command: list[str]) -> dict[str, Any]:
    started_at = utc_now()
    result = subprocess.run(command, cwd=ROOT)
    return {
        "command": command,
        "started_at": started_at,
        "finished_at": utc_now(),
        "returncode": result.returncode,
    }


def load_metrics_rows() -> list[dict[str, Any]]:
    if not METRICS_TABLE.exists():
        return []
    try:
        rows = json.loads(METRICS_TABLE.read_text())
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def matching_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    normalized_label = safe_name(label).lower()
    label_parts = [part for part in re.split(r"[_\W]+", normalized_label) if part]
    matches: list[dict[str, Any]] = []
    for row in rows:
        run_name = str(row.get("run_name", ""))
        config_stem = Path(str(row.get("config", ""))).stem
        haystack = f"{run_name} {config_stem}".lower()
        if normalized_label in haystack or all(part in haystack for part in label_parts):
            matches.append(row)
    return matches


def pre_gmm_candidates() -> list[dict[str, Any]]:
    rows = load_metrics_rows()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        config_stem = Path(str(row.get("config", ""))).stem
        run_name = str(row.get("run_name", ""))
        text = f"{config_stem} {run_name}"
        if "realonly_993" not in text and "transfer_993" not in text:
            continue
        candidates.append(
            {
                "run_name": run_name,
                "config": row.get("config"),
                "status": row.get("status"),
                "observed_accuracy": row.get("observed_accuracy"),
                "artifacts_usable": row.get("artifacts_usable"),
                "log_path": row.get("log_path"),
                "checkpoint_path": row.get("checkpoint_path"),
                "figure_action": figure_action(row),
            }
        )
    return candidates


def figure_action(row: dict[str, Any]) -> str:
    if row.get("observed_accuracy") is None or row.get("status") == "pending":
        return "wait for completed checker outputs"
    if row.get("artifacts_usable") is False:
        return "blocked until artifacts are usable"
    return "draw Figure 3-6 diagnostics"


def send_wandb_alert(title: str, text: str, label: str) -> dict[str, Any]:
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"sent": False, "error": f"wandb import failed: {exc}"}

    try:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)

        init_kwargs: dict[str, Any] = {
            "project": os.environ.get("WANDB_ALERT_PROJECT", os.environ.get("WANDB_PROJECT", "viscnet-rebuild")),
            "job_type": "notification",
            "name": f"notify_{safe_name(label)}",
            "reinit": True,
        }
        entity = os.environ.get("WANDB_ALERT_ENTITY") or os.environ.get("WANDB_ENTITY")
        if entity:
            init_kwargs["entity"] = entity

        run = wandb.init(**init_kwargs)
        wandb.alert(title=title, text=text)
        run.finish()
        return {"sent": True, "project": init_kwargs["project"], "entity": init_kwargs.get("entity")}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"sent": False, "error": str(exc)}


def output_paths(summary_path: Path) -> dict[str, str]:
    return {
        "checker_report": relative_or_str(CHECKER_REPORT),
        "analyzer_report": relative_or_str(ANALYZER_REPORT),
        "metrics_table": relative_or_str(METRICS_TABLE),
        "data_efficiency_csv": relative_or_str(DATA_EFFICIENCY_CSV),
        "data_efficiency_json": relative_or_str(DATA_EFFICIENCY_JSON),
        "data_efficiency_plot": relative_or_str(DATA_EFFICIENCY_PLOT),
        "summary": relative_or_str(summary_path),
    }


def relative_or_str(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def row_summary(row: dict[str, Any]) -> str:
    accuracy = row.get("observed_accuracy")
    accuracy_text = "pending" if accuracy is None else f"{float(accuracy):.4f}"
    return (
        f"- `{row.get('run_name')}`: status=`{row.get('status')}`, "
        f"accuracy={accuracy_text}, target={row.get('target_accuracy')}, "
        f"log=`{row.get('log_path')}`"
    )


def write_status_summary(
    summary_path: Path,
    label: str,
    failed_count: int,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    paths: dict[str, str],
) -> None:
    lines = [
        f"# Post-Training Summary: {label}",
        "",
        f"- Finished at: `{utc_now()}`",
        f"- Failed post-training commands: `{failed_count}`",
        f"- Checker report: `{paths['checker_report']}`",
        f"- Analyzer report: `{paths['analyzer_report']}`",
        f"- Metrics table: `{paths['metrics_table']}`",
        "",
        "## Matching Runs",
        "",
    ]
    if rows:
        lines.extend(row_summary(row) for row in rows)
    else:
        lines.append("- No metrics-table row matched this label.")
    lines.extend(["", "## Eligible 993 Pre-GMM Figure Candidates", ""])
    if candidates:
        lines.extend(
            f"- `{row.get('run_name')}`: {row.get('figure_action')}; log=`{row.get('log_path')}`" for row in candidates
        )
    else:
        lines.append("- None.")
    summary_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="rebuild_training", help="Label used for the marker file and alert name.")
    parser.add_argument(
        "--skip-data-efficiency-plot", action="store_true", help="Do not regenerate the data-efficiency plot."
    )
    parser.add_argument("--wandb-alert", action="store_true", help="Send a best-effort W&B alert after post-analysis.")
    parser.add_argument("--alert-title", default="", help="Override W&B alert title.")
    parser.add_argument("--alert-text", default="", help="Override W&B alert body.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    NOTIFY_ROOT.mkdir(parents=True, exist_ok=True)

    commands = [
        [sys.executable, "scripts/check_rebuild_results.py"],
        [sys.executable, "scripts/analyze_rebuild_results.py"],
    ]
    if not args.skip_data_efficiency_plot:
        commands.append([sys.executable, "scripts/plot_data_efficiency.py"])

    command_results = [run_command(command) for command in commands]
    failed = [item for item in command_results if item["returncode"] != 0]

    label = safe_name(args.label)
    summary_path = NOTIFY_ROOT / f"{label}_summary.md"
    rows = load_metrics_rows()
    matched = matching_rows(rows, args.label)
    candidates = pre_gmm_candidates()
    paths = output_paths(summary_path)
    write_status_summary(summary_path, args.label, len(failed), matched, candidates, paths)
    title = args.alert_title or f"ViscNet training finished: {args.label}"
    text = args.alert_text or (
        "Post-training checker/analyzer completed. "
        f"Failed post commands: {len(failed)}. "
        f"Eligible 993 pre-GMM figure candidates: {len(candidates)}."
    )

    alert_result: dict[str, Any] = {"sent": False, "reason": "disabled"}
    if args.wandb_alert:
        alert_result = send_wandb_alert(title, text, args.label)

    marker = {
        "label": args.label,
        "finished_at": utc_now(),
        "commands": command_results,
        "failed_command_count": len(failed),
        "output_paths": paths,
        "matching_runs": [
            {
                "run_name": row.get("run_name"),
                "config": row.get("config"),
                "status": row.get("status"),
                "observed_accuracy": row.get("observed_accuracy"),
                "target_accuracy": row.get("target_accuracy"),
                "log_path": row.get("log_path"),
                "checkpoint_path": row.get("checkpoint_path"),
            }
            for row in matched
        ],
        "pre_gmm_figure_candidates": candidates,
        "wandb_alert": alert_result,
    }
    marker_path = NOTIFY_ROOT / f"{label}_complete.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    print(f"Wrote {marker_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
