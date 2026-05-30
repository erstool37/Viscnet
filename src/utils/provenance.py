"""W&B project resolution and launch provenance helpers."""

from __future__ import annotations

import copy
import os
import shlex
import subprocess
from pathlib import Path


DEFAULT_REBUILD_PROJECT = "re-rebuild-viscnet"
LEGACY_REBUILD_PROJECT = "viscnet-rebuild"


def is_rebuild_config_path(config_path):
    normalized = Path(config_path).as_posix()
    return normalized.startswith("configs/rebuild/") or "/configs/rebuild/" in normalized


def resolve_wandb_project(config, config_path, env=None):
    env = os.environ if env is None else env
    requested_project = env.get("WANDB_PROJECT") or config.get("project")
    if is_rebuild_config_path(config_path) and requested_project == LEGACY_REBUILD_PROJECT:
        return DEFAULT_REBUILD_PROJECT
    return requested_project


def _git_output(args, cwd=None):
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def collect_launch_metadata(config_path, argv=None, cwd=None):
    argv = [] if argv is None else list(argv)
    status = _git_output(["status", "--porcelain"], cwd=cwd)
    return {
        "git_commit": _git_output(["rev-parse", "HEAD"], cwd=cwd),
        "git_branch": _git_output(["branch", "--show-current"], cwd=cwd),
        "git_dirty": bool(status),
        "config_path": str(config_path),
        "launch_command": " ".join(shlex.quote(part) for part in argv),
    }


def build_wandb_config(config, metadata, project=None):
    wandb_config = copy.deepcopy(config)
    if project is not None:
        wandb_config["project"] = project
    wandb_config["provenance"] = dict(metadata)
    return wandb_config
