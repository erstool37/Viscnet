#!/usr/bin/env python3
"""Wait for a tmux training session, then run post-training analysis once."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tmux_has_session(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def wait_for_signal(signal: str) -> int:
    return subprocess.run(["tmux", "wait-for", signal], cwd=ROOT).returncode


def wait_for_session_exit(session: str, poll_interval: int, allow_missing: bool) -> int:
    if not tmux_has_session(session):
        if allow_missing:
            return 0
        print(f"tmux session does not exist: {session}", file=sys.stderr)
        return 2

    print(f"Watching tmux session `{session}`. No output will be emitted until it exits.")
    while tmux_has_session(session):
        time.sleep(poll_interval)
    return 0


def run_post_analysis(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "scripts/post_rebuild_training.py",
        "--label",
        args.label,
    ]
    if args.skip_data_efficiency_plot:
        command.append("--skip-data-efficiency-plot")
    if args.wandb_alert:
        command.append("--wandb-alert")
    if args.alert_title:
        command.extend(["--alert-title", args.alert_title])
    if args.alert_text:
        command.extend(["--alert-text", args.alert_text])
    return subprocess.run(command, cwd=ROOT).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", help="tmux session to watch until it exits.")
    parser.add_argument("--signal", help="tmux wait-for signal name. Prefer this for newly launched instrumented jobs.")
    parser.add_argument(
        "--label", default="tmux_training", help="Label for post-analysis marker and optional W&B alert."
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60, help="Seconds between tmux session checks when --signal is not used."
    )
    parser.add_argument(
        "--allow-missing", action="store_true", help="Treat a missing tmux session as already complete."
    )
    parser.add_argument(
        "--no-post-analysis", action="store_true", help="Only wait; do not run checker/analyzer afterward."
    )
    parser.add_argument(
        "--skip-data-efficiency-plot", action="store_true", help="Do not regenerate data-efficiency plot."
    )
    parser.add_argument("--wandb-alert", action="store_true", help="Send a best-effort W&B alert after post-analysis.")
    parser.add_argument("--alert-title", default="", help="Override W&B alert title.")
    parser.add_argument("--alert-text", default="", help="Override W&B alert body.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.session and not args.signal:
        print("Either --session or --signal is required.", file=sys.stderr)
        return 2

    if args.signal:
        print(f"Waiting for tmux signal `{args.signal}`.")
        wait_code = wait_for_signal(args.signal)
    else:
        wait_code = wait_for_session_exit(args.session, args.poll_interval, args.allow_missing)

    if wait_code != 0:
        return wait_code
    if args.no_post_analysis:
        return 0
    return run_post_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
