#!/usr/bin/env python3
"""Fail fast if a queued no-RPM config can pass RPM conditioning to the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="YAML config paths to verify")
    parser.add_argument("--json", action="store_true", help="Emit a JSON report")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: config did not parse to a mapping")
    return payload


def verify_config(path: Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    embeddings = cfg.get("model", {}).get("embeddings", {})
    training = cfg.get("training", {})
    rpm_bool = embeddings.get("rpm_bool")
    curr_bool = bool(training.get("curr_bool", False))
    curr_ckpt = str(training.get("curr_ckpt", ""))
    name = cfg.get("name", path.stem)
    errors = []
    if rpm_bool is not False:
        errors.append(f"{path}: model.embeddings.rpm_bool must be false, got {rpm_bool!r}")
    if curr_bool and "no_rpm" not in curr_ckpt:
        errors.append(f"{path}: curr_ckpt for no-RPM transfer must reference no_rpm weights, got {curr_ckpt!r}")
    if not str(name).startswith(("repro_", "dual_pattern_", "allnew_", "crosspat_")):
        errors.append(f"{path}: unexpected run name {name!r}")
    return {
        "path": str(path),
        "name": name,
        "rpm_bool": rpm_bool,
        "curr_bool": curr_bool,
        "curr_ckpt": curr_ckpt,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    results = [verify_config(Path(path)) for path in args.configs]
    errors = [error for result in results for error in result["errors"]]

    if args.json:
        print(json.dumps({"ok": not errors, "results": results}, indent=2))
    else:
        for result in results:
            status = "ok" if not result["errors"] else "failed"
            print(f"{status}: {result['path']} rpm_bool={result['rpm_bool']!r}")
        for error in errors:
            print(error)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
