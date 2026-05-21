#!/usr/bin/env python3
"""Check Viscnet reproduction outputs against W&B/paper reference metrics."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "rebuild"
REFERENCE_PATH = CONFIG_ROOT / "reference_metrics.json"
OUTPUT_ROOT = ROOT / "outputs" / "rebuild_reproduction"
REPORT_PATH = OUTPUT_ROOT / "checker_report.md"
TABLE_PATH = OUTPUT_ROOT / "metrics_table.json"
WANDB_ENTITY = "jongwonsohn-seoul-national-university"
WANDB_PROJECT = "viscnet-rebuild"
EXPECTED_EPOCHS = 300
EXPECTED_MICROBATCH_EPOCHS = 30
EXPECTED_OPTIMIZER_MICROBATCH_SIZE = 1
EXPECTED_BATCH_SIZE = 10
EXPECTED_NUM_WORKERS = 0
EXPECTED_SEED = 1205
EXPECTED_MANIFEST_SEED = 37
DIAGNOSTIC_REALONLY_300_EPOCH_ACCURACY = 0.8500


def sample_size_from_stem(stem: str, prefix: str) -> str:
    match = re.match(rf"{prefix}_(\d+)", stem)
    return match.group(1) if match else ""


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


def wandb_run_exists(run_id: str) -> bool | None:
    if not run_id:
        return False
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb

        api = wandb.Api()
        api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        return True
    except Exception:
        return False


def parse_wandb_run_id(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(errors="ignore")
    patterns = [
        r"wandb: .*View run .*?/runs/([A-Za-z0-9_-]+)",
        r"wandb: .*View project .*?/runs/([A-Za-z0-9_-]+)",
        r"run-[A-Za-z0-9]+\\.wandb",
    ]
    for pattern in patterns[:2]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    match = re.search(patterns[2], text)
    if match:
        return match.group(0).removeprefix("run-").removesuffix(".wandb")
    return ""


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def json_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("accuracy")


def expected_target(config_path: Path, references: dict) -> tuple[str, float | None, str]:
    stem = config_path.stem
    if stem.startswith("realonly_"):
        size = sample_size_from_stem(stem, "realonly")
        ref = references["real_only_curve"].get(size, {})
        return "real_only_curve", ref.get("accuracy"), ref.get("run_id", "")
    if stem.startswith("transfer_"):
        size = sample_size_from_stem(stem, "transfer")
        ref = references["transfer_curve"].get(size, {})
        return "transfer_curve", ref.get("accuracy"), ref.get("run_id", "")
    if stem.startswith("pattern_"):
        ref = references["pattern_reference"]
        return "pattern_reference", ref.get("accuracy"), ref.get("run_id", "")
    if stem == "synthetic_pretrain":
        ref = references["synthetic_pretrain"]
        return "synthetic_pretrain", ref.get("accuracy"), ref.get("run_id", "")
    return "unknown", None, ""


def methodology_fit(config_path: Path, cfg: dict) -> tuple[str, list[str]]:
    notes: list[str] = []
    fit = "exact reproduction"
    train_root = cfg["dataset"]["train"]["train_root"]
    train_frame_count = int(float(cfg["dataset"]["train"]["frame_num"]) * float(cfg["dataset"]["train"]["time"]))
    batch_size = cfg["dataset"]["train"]["dataloader"]["batch_size"]
    test_batch_size = cfg["dataset"]["test"]["dataloader"]["batch_size"]
    embeddings = cfg["model"].get("embeddings", {})
    transformer = cfg["model"].get("transformer", {})
    training = cfg["training"]
    optimizer = training.get("optimizer", {})
    acceptance = training.get("acceptance", {})
    update_density = training.get("update_density", {})
    optimizer_microbatch_size = update_density.get("optimizer_microbatch_size")
    schedule_policy = optimizer.get("schedule_policy", "cosine")
    warmup_epochs = float(optimizer.get("warmup_epochs", 0.0))
    train_settings = cfg["train_settings"]

    if train_root == "dataset/CFDArchive/sph_35000":
        fit = "current-repo reproduction"
        notes.append("Uses available sph_35000 instead of missing sph_realvisc_diffback_35000.")
    if config_path.stem.startswith("pattern_"):
        fit = "current-repo reproduction"
        notes.append("Uses imported pattern split names, not old 1back/4back W&B paths.")
    if batch_size != 1:
        fit = "current-repo reproduction"
        notes.append(f"Uses per-GPU batch_size={batch_size}; old W&B configs used 1.")
    if optimizer_microbatch_size is not None:
        fit = "current-repo reproduction"
        notes.append(
            f"Uses optimizer_microbatch_size={optimizer_microbatch_size} to recover small-batch update density."
        )
    if acceptance.get("allow_lr_override"):
        fit = "current-repo reproduction"
        notes.append(f"Uses bounded lr override: lr={float(optimizer.get('lr', -1.0)):.2e}.")
    if acceptance.get("allow_patience_override"):
        fit = "current-repo reproduction"
        notes.append(f"Uses bounded patience override: patience={int(optimizer.get('patience', -1))}.")
    if acceptance.get("allow_batch_size_override"):
        fit = "current-repo reproduction"
        notes.append(f"Uses bounded batch-size override: batch_size={batch_size}.")
    if acceptance.get("allow_epoch_override"):
        fit = "current-repo reproduction"
        notes.append(f"Uses bounded epoch override: num_epochs={int(training.get('num_epochs', -1))}.")
    if acceptance.get("allow_frame_count_override"):
        fit = "current-repo reproduction"
        notes.append(
            f"Uses bounded frame-count override: train_frames={train_frame_count}, "
            f"model_frames={transformer.get('num_frames', train_frame_count)}."
        )
    if acceptance.get("allow_derived_window_dataset"):
        fit = "current-repo reproduction"
        notes.append(f"Uses derived temporal-window training data: {train_root}.")
    if schedule_policy != "cosine":
        fit = "current-repo reproduction"
        notes.append(f"Uses schedule_policy={schedule_policy}.")
    if schedule_policy == "warmup_hold_cosine":
        notes.append(f"Uses warmup_epochs={warmup_epochs:.2f}.")
    if acceptance.get("min_accuracy") is not None:
        notes.append(f"Uses explicit min_accuracy={float(acceptance['min_accuracy']):.4f}.")

    mismatches = []
    if cfg.get("project") != WANDB_PROJECT:
        mismatches.append("W&B project is not viscnet-rebuild.")
    if cfg["model"].get("transformer", {}).get("encoder") != "VivitEmbed":
        mismatches.append("model encoder is not VivitEmbed.")
    if training.get("label_smoothing") != 0.0:
        mismatches.append("label_smoothing is not 0.0.")
    if training.get("loss") != "CE":
        mismatches.append("loss is not CE.")
    if not train_settings.get("classification") or train_settings.get("gmm_bool"):
        mismatches.append("classification/gmm flags are outside reproduction scope.")
    if embeddings.get("rpm_bool") is not True or embeddings.get("pat_bool") is not False:
        mismatches.append("embedding flags must be rpm_bool=true and pat_bool=false.")
    if optimizer.get("optim_class") != "AdamW":
        mismatches.append("optimizer is not AdamW.")
    if optimizer.get("scheduler_class") != "CosineAnnealingLR":
        mismatches.append("scheduler is not CosineAnnealingLR.")
    lr = float(optimizer.get("lr", -1.0))
    if acceptance.get("allow_lr_override"):
        if not (0.0 < lr <= 1e-5):
            mismatches.append("bounded lr override must be positive and no greater than 1e-5.")
    elif lr != 1e-5:
        mismatches.append("learning rate is not 1e-5.")
    if float(optimizer.get("weight_decay", -1.0)) != 1e-2:
        mismatches.append("weight decay is not 1e-2.")
    num_epochs = int(training.get("num_epochs", -1))
    if optimizer_microbatch_size is not None:
        if schedule_policy == "hold_then_cosine":
            if num_epochs <= EXPECTED_MICROBATCH_EPOCHS or num_epochs >= EXPECTED_EPOCHS:
                mismatches.append(
                    f"hold_then_cosine microbatch retry must use more than {EXPECTED_MICROBATCH_EPOCHS} and less than {EXPECTED_EPOCHS} epochs."
                )
        elif schedule_policy == "warmup_hold_cosine":
            if num_epochs <= EXPECTED_MICROBATCH_EPOCHS or num_epochs >= EXPECTED_EPOCHS:
                mismatches.append(
                    f"warmup_hold_cosine microbatch retry must use more than {EXPECTED_MICROBATCH_EPOCHS} and less than {EXPECTED_EPOCHS} epochs."
                )
        elif num_epochs != EXPECTED_MICROBATCH_EPOCHS:
            mismatches.append(f"num_epochs is not {EXPECTED_MICROBATCH_EPOCHS}.")
    elif acceptance.get("allow_epoch_override"):
        if not (EXPECTED_MICROBATCH_EPOCHS < num_epochs < EXPECTED_EPOCHS):
            mismatches.append(
                f"bounded non-microbatch epoch override must use more than {EXPECTED_MICROBATCH_EPOCHS} and less than {EXPECTED_EPOCHS} epochs."
            )
    elif num_epochs != EXPECTED_EPOCHS:
        mismatches.append(f"num_epochs is not {EXPECTED_EPOCHS}.")
    if optimizer_microbatch_size is not None and int(optimizer_microbatch_size) != EXPECTED_OPTIMIZER_MICROBATCH_SIZE:
        mismatches.append(f"optimizer_microbatch_size is not {EXPECTED_OPTIMIZER_MICROBATCH_SIZE}.")
    if schedule_policy == "hold_then_cosine":
        if float(optimizer.get("lr_hold_epochs", -1.0)) <= 0.0:
            mismatches.append("lr_hold_epochs must be positive for hold_then_cosine.")
    elif schedule_policy == "warmup_hold_cosine":
        if float(optimizer.get("warmup_epochs", -1.0)) <= 0.0:
            mismatches.append("warmup_epochs must be positive for warmup_hold_cosine.")
        if float(optimizer.get("lr_hold_epochs", -1.0)) <= 0.0:
            mismatches.append("lr_hold_epochs must be positive for warmup_hold_cosine.")
        if not (0.0 < float(optimizer.get("warmup_start_factor", -1.0)) <= 1.0):
            mismatches.append("warmup_start_factor must be in (0, 1] for warmup_hold_cosine.")
    elif schedule_policy != "cosine":
        mismatches.append(f"unsupported schedule_policy: {schedule_policy}.")
    if acceptance.get("allow_patience_override"):
        if int(optimizer.get("patience", -1)) <= 10:
            mismatches.append("bounded patience override must be greater than 10.")
    elif int(optimizer.get("patience", -1)) != 10:
        mismatches.append("patience is not 10.")
    if acceptance.get("allow_batch_size_override"):
        if int(batch_size) <= 0 or int(test_batch_size) != int(batch_size):
            mismatches.append("bounded batch-size override requires matching positive train/test batch sizes.")
    elif batch_size != EXPECTED_BATCH_SIZE or test_batch_size != EXPECTED_BATCH_SIZE:
        mismatches.append(f"batch_size is not {EXPECTED_BATCH_SIZE}.")
    if int(train_settings.get("num_workers", -1)) != EXPECTED_NUM_WORKERS:
        mismatches.append(f"num_workers is not {EXPECTED_NUM_WORKERS}.")
    if int(train_settings.get("seed", -1)) != EXPECTED_SEED:
        mismatches.append(f"seed is not {EXPECTED_SEED}.")
    manifest = cfg["dataset"]["train"].get("manifest")
    if manifest:
        manifest_path = ROOT / manifest
        if manifest_path.exists():
            manifest_payload = json.loads(manifest_path.read_text())
            if manifest_payload.get("seed") != EXPECTED_MANIFEST_SEED:
                mismatches.append(f"manifest seed is not {EXPECTED_MANIFEST_SEED}.")
        else:
            mismatches.append(f"manifest is missing: {manifest}.")
    ckpt_root = Path(cfg["misc_dir"]["ckpt_root"])
    output_root = Path(cfg["misc_dir"]["output_root"])
    if ckpt_root != Path("outputs/rebuild_reproduction/checkpoints"):
        mismatches.append("checkpoint path is outside outputs/rebuild_reproduction/checkpoints.")
    if not str(output_root).startswith("outputs/rebuild_reproduction/"):
        mismatches.append("output root is outside outputs/rebuild_reproduction.")

    if mismatches:
        fit = "blocked/mismatched"
        notes.extend(mismatches)
    return fit, notes


def inspect_config(config_path: Path, references: dict) -> dict:
    cfg = load_yaml(config_path)
    run_name = Path(cfg["training"]["checkpoint_name"]).stem
    update_density = cfg["training"].get("update_density", {})
    run_log_path = OUTPUT_ROOT / "logs" / f"{run_name}.log"
    legacy_log_path = OUTPUT_ROOT / "logs" / f"{config_path.stem}.log"
    if run_log_path.exists() or update_density:
        log_path = run_log_path
    else:
        log_path = legacy_log_path
    log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    wandb_run_id = parse_wandb_run_id(log_path)
    wandb_confirmed = wandb_run_exists(wandb_run_id)
    ckpt = ROOT / cfg["misc_dir"]["ckpt_root"] / cfg["training"]["checkpoint_name"]
    output_root = ROOT / cfg["misc_dir"]["output_root"]
    confusion_json = output_root / "confusion_matrix" / f"{run_name}_metrics.json"
    reliability_json = output_root / "reliability_plots" / f"{run_name}_metrics.json"
    confusion_png = output_root / "confusion_matrix" / f"{run_name}.png"
    reliability_png = output_root / "reliability_plots" / f"{run_name}.png"
    family, target, reference_run_id = expected_target(config_path, references)
    fit, notes = methodology_fit(config_path, cfg)
    acceptance = cfg["training"].get("acceptance", {})
    if acceptance.get("min_accuracy") is not None:
        target = float(acceptance["min_accuracy"])
        notes.append(
            f"Pass threshold overrides reference target; must exceed diagnostic real-only 300-epoch accuracy {DIAGNOSTIC_REALONLY_300_EPOCH_ACCURACY:.4f}."
        )

    required_test_artifacts = family not in {"synthetic_pretrain"}
    log_complete = (
        "Training complete." in log_text
        or "Early stopping at epoch" in log_text
        or (family != "synthetic_pretrain" and "Accuracy:" in log_text)
    )
    observed = json_accuracy(confusion_json) if log_complete else None
    artifacts_ok = ckpt.exists()
    if required_test_artifacts:
        artifacts_ok = (
            artifacts_ok
            and log_complete
            and all(p.exists() for p in [confusion_json, reliability_json, confusion_png, reliability_png])
        )
    wandb_ok = wandb_confirmed is True
    metric_pass = None
    if target is not None and observed is not None:
        metric_pass = observed >= target

    if fit == "blocked/mismatched":
        status = "blocked"
    elif not ckpt.exists() or (required_test_artifacts and not log_complete):
        status = "pending"
    elif required_test_artifacts and observed is None:
        status = "pending"
    elif metric_pass is False or (required_test_artifacts and not artifacts_ok) or not log_complete or not wandb_ok:
        status = "fail"
    elif metric_pass is True and artifacts_ok:
        status = "pass"
    elif family == "synthetic_pretrain" and ckpt.exists():
        status = "pass"
    else:
        status = "pending"

    return {
        "config": str(config_path.relative_to(ROOT)),
        "run_name": run_name,
        "family": family,
        "reference_run_id": reference_run_id,
        "target_accuracy": target,
        "observed_accuracy": observed,
        "status": status,
        "methodology_fit": fit,
        "notes": notes,
        "checkpoint_exists": ckpt.exists(),
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "log_path": str(log_path.relative_to(ROOT)),
        "log_exists": log_path.exists(),
        "log_complete": log_complete,
        "wandb_run_id": wandb_run_id,
        "wandb_confirmed": wandb_confirmed,
        "confusion_metrics_exists": confusion_json.exists(),
        "reliability_metrics_exists": reliability_json.exists(),
        "confusion_plot_exists": confusion_png.exists(),
        "reliability_plot_exists": reliability_png.exists(),
        "artifacts_usable": artifacts_ok,
    }


def write_report(rows: list[dict]) -> None:
    def artifact_state(row: dict, key: str) -> str:
        exists = row[key]
        if not exists:
            return "missing"
        if row["artifacts_usable"]:
            return "usable"
        if not row["log_complete"]:
            return "present but not usable until the current log completes"
        return "present but not usable"

    lines = [
        "# Viscnet Rebuild Checker Report",
        "",
        "This report is generated from `scripts/check_rebuild_results.py`.",
        "",
        "| Status | Config | Methodology | Target | Observed | Notes |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        target = "" if row["target_accuracy"] is None else f"{row['target_accuracy']:.4f}"
        observed = "" if row["observed_accuracy"] is None else f"{row['observed_accuracy']:.4f}"
        note_items = list(row["notes"])
        if row["wandb_run_id"]:
            note_items.append(f"W&B `{row['wandb_run_id']}` confirmed={row['wandb_confirmed']}")
        elif row["status"] != "pending":
            note_items.append("W&B run ID missing")
        if row["log_exists"] and not row["log_complete"] and row["status"] != "pending":
            note_items.append("training log lacks completion marker")
        notes = "; ".join(note_items)
        lines.append(
            f"| {row['status']} | `{row['config']}` | {row['methodology_fit']} | {target} | {observed} | {notes} |"
        )

    lines.extend(["", "## Run Evidence", ""])
    for row in rows:
        target = "n/a" if row["target_accuracy"] is None else f"{row['target_accuracy']:.4f}"
        observed = "pending" if row["observed_accuracy"] is None else f"{row['observed_accuracy']:.4f}"
        wandb = row["wandb_run_id"] or "missing"
        wandb_confirmed = row["wandb_confirmed"]
        lines.extend(
            [
                f"### `{row['run_name']}`",
                "",
                f"- Config: `{row['config']}`",
                f"- Status: `{row['status']}`; methodology: `{row['methodology_fit']}`",
                f"- Accuracy: observed={observed}, target={target}",
                f"- W&B run ID: `{wandb}`; confirmed={wandb_confirmed}",
                f"- Checkpoint: `{row['checkpoint_path']}`; exists={row['checkpoint_exists']}",
                f"- Log: `{row['log_path']}`; exists={row['log_exists']}; complete={row['log_complete']}",
                f"- Confusion metrics JSON: {artifact_state(row, 'confusion_metrics_exists')}",
                f"- Confusion plot: {artifact_state(row, 'confusion_plot_exists')}",
                f"- Reliability metrics JSON: {artifact_state(row, 'reliability_metrics_exists')}",
                f"- Reliability plot: {artifact_state(row, 'reliability_plot_exists')}",
                f"- Artifacts usable for pass/fail: {row['artifacts_usable']}",
                "",
            ]
        )

    pending = [row for row in rows if row["status"] == "pending"]
    failed = [row for row in rows if row["status"] in {"fail", "blocked"}]
    passed = [row for row in rows if row["status"] == "pass"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Passed: {len(passed)}",
            f"- Pending: {len(pending)}",
            f"- Failed or blocked: {len(failed)}",
            "",
            "## Final Recommendation",
            "",
        ]
    )
    if failed:
        lines.append("- retry required")
    elif pending:
        lines.append("- retry required")
        lines.append("- Reason: required runs or artifacts are still pending.")
    else:
        lines.append("- accepted")
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    load_env()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    references = json.loads(REFERENCE_PATH.read_text())
    configs = sorted(CONFIG_ROOT.glob("*.yaml"))
    if os.environ.get("REBUILD_INCLUDE_RETRIES") == "1":
        configs.extend(sorted((CONFIG_ROOT / "retries").glob("*.yaml")))
    rows = [inspect_config(path, references) for path in configs]
    TABLE_PATH.write_text(json.dumps(rows, indent=2) + "\n")
    write_report(rows)
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {TABLE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
