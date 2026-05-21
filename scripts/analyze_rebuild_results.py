#!/usr/bin/env python3
"""Generate a post-verifier analysis report for Viscnet rebuild runs."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "rebuild_reproduction"
CHECKER_TABLE = OUTPUT_ROOT / "metrics_table.json"
CHECKER_REPORT = OUTPUT_ROOT / "checker_report.md"
ANALYZER_REPORT = OUTPUT_ROOT / "analyzer_report.md"
REFERENCE_PATH = ROOT / "configs" / "rebuild" / "reference_metrics.json"
WANDB_ENTITY = "jongwonsohn-seoul-national-university"


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


def read_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def run_output_root(row: dict) -> Path:
    run_name = row["run_name"]
    return OUTPUT_ROOT / run_name


def metrics_for(row: dict, subdir: str) -> dict:
    if row.get("artifacts_usable") is False:
        return {}
    path = run_output_root(row) / subdir / f"{row['run_name']}_metrics.json"
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def parse_losses(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="ignore")
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+results\s+-\s+Train Loss:\s+([0-9.]+)\s+"
        r"Validation Loss:\s+([0-9.]+)\s+-\s+LR:\s+([0-9.eE+-]+)"
    )
    rows = []
    for match in pattern.finditer(text):
        rows.append(
            {
                "epoch": int(match.group(1)),
                "total_epochs": int(match.group(2)),
                "train_loss": float(match.group(3)),
                "val_loss": float(match.group(4)),
                "lr": float(match.group(5)),
            }
        )
    return rows


def training_horizon_assessment(losses: list[dict]) -> str:
    if len(losses) < 4:
        return "insufficient loss history for horizon assessment"
    first = losses[0]
    last = losses[-1]
    best = min(losses, key=lambda item: item["val_loss"])
    total_epochs = last["total_epochs"]
    recent = losses[-5:] if len(losses) >= 5 else losses
    recent_delta = recent[0]["val_loss"] - last["val_loss"]
    initial_lr = first["lr"]
    lr_ratio = last["lr"] / initial_lr if initial_lr > 0 else 0.0
    best_is_late = best["epoch"] >= max(1, int(total_epochs * 0.8))
    lr_exhausted = lr_ratio <= 0.1
    still_improving = recent_delta > 0.01
    if lr_exhausted and (best_is_late or still_improving):
        return (
            "likely under-trained for current batch size: LR is exhausted while "
            f"best validation is late/recently improving (best_epoch={best['epoch']}, "
            f"last_epoch={last['epoch']}, recent_val_delta={recent_delta:.4f}, "
            f"lr_ratio={lr_ratio:.4f})"
        )
    if best_is_late:
        return (
            "watch horizon: best validation occurs late in the schedule "
            f"(best_epoch={best['epoch']}, last_epoch={last['epoch']})"
        )
    return (
        "horizon not currently flagged by loss/LR heuristic "
        f"(best_epoch={best['epoch']}, last_epoch={last['epoch']}, lr_ratio={lr_ratio:.4f})"
    )


def worst_classes(confusion: dict, limit: int = 3) -> list[tuple[int, float, int]]:
    labels = confusion.get("labels") or []
    per_class = confusion.get("per_class_accuracy") or []
    support = confusion.get("support") or []
    rows = []
    for idx, label in enumerate(labels):
        rows.append((int(label), float(per_class[idx]), int(support[idx]) if idx < len(support) else 0))
    rows.sort(key=lambda item: (item[1], -item[2]))
    return rows[:limit]


def largest_confusions(confusion: dict, limit: int = 5) -> list[tuple[int, int, int]]:
    labels = confusion.get("labels") or []
    counts = confusion.get("confusion_matrix_counts") or []
    pairs = []
    for i, row in enumerate(counts):
        for j, count in enumerate(row):
            if i != j and count:
                true_label = int(labels[i]) if i < len(labels) else i
                pred_label = int(labels[j]) if j < len(labels) else j
                pairs.append((true_label, pred_label, int(count)))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:limit]


def fetch_reference_summaries(references: dict) -> dict[str, dict]:
    if not os.environ.get("WANDB_API_KEY"):
        return {}
    try:
        import wandb

        api = wandb.Api()
    except Exception:
        return {}

    out: dict[str, dict] = {}
    run_paths = set()
    for family in ["real_only_curve", "transfer_curve"]:
        for ref in references.get(family, {}).values():
            if ref.get("project") and ref.get("run_id"):
                run_paths.add((ref["project"], ref["run_id"]))
    pattern = references.get("pattern_reference", {})
    if pattern.get("project") and pattern.get("run_id"):
        run_paths.add((pattern["project"], pattern["run_id"]))

    for project, run_id in sorted(run_paths):
        key = f"{project}/{run_id}"
        try:
            run = api.run(f"{WANDB_ENTITY}/{project}/{run_id}")
            out[key] = {
                "name": run.name,
                "state": run.state,
                "summary": dict(run.summary),
                "config_keys": sorted(list(run.config.keys()))[:40],
            }
        except Exception as exc:
            out[key] = {"error": str(exc)}
    return out


def data_efficiency_rows(rows: list[dict]) -> list[str]:
    by_size: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        stem = Path(row["config"]).stem
        if stem.startswith("realonly_"):
            size = sample_size_from_stem(stem, "realonly")
            if size:
                by_size[size]["realonly"] = row
        if stem.startswith("transfer_"):
            size = sample_size_from_stem(stem, "transfer")
            if size:
                by_size[size]["transfer"] = row

    lines = [
        "| Samples | Real-only | Transfer | Transfer Gain | Target Status |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for size in sorted(by_size, key=lambda value: int(value)):
        real = by_size[size].get("realonly", {})
        transfer = by_size[size].get("transfer", {})
        real_acc = real.get("observed_accuracy")
        transfer_acc = transfer.get("observed_accuracy")
        gain = ""
        if real_acc is not None and transfer_acc is not None:
            gain = f"{transfer_acc - real_acc:+.4f}"
        lines.append(
            "| {size} | {real_acc} | {transfer_acc} | {gain} | {status} |".format(
                size=size,
                real_acc="" if real_acc is None else f"{real_acc:.4f}",
                transfer_acc="" if transfer_acc is None else f"{transfer_acc:.4f}",
                gain=gain,
                status=f"real={real.get('status', '')}, transfer={transfer.get('status', '')}",
            )
        )
    return lines


def pre_gmm_figure_candidate(row: dict) -> bool:
    text = f"{Path(row.get('config', '')).stem} {row.get('run_name', '')}"
    return "realonly_993" in text or "transfer_993" in text


def pre_gmm_figure_trigger_rows(rows: list[dict]) -> list[str]:
    candidates = [row for row in rows if pre_gmm_figure_candidate(row)]
    non_993_sizes = ["300", "400", "500", "600", "700", "800", "900"]
    skipped = []
    for row in rows:
        stem = Path(row.get("config", "")).stem
        if stem.startswith(("realonly_", "transfer_")) and any(
            stem.startswith(f"realonly_{size}") or stem.startswith(f"transfer_{size}") for size in non_993_sizes
        ):
            skipped.append(stem)

    lines = [
        "- Trigger rule: draw pre-GMM paper-style Figure 3-6 diagnostics only for 993-sample real-only or transfer classification results.",
        "- Excluded: data-efficiency runs with 300, 400, 500, 600, 700, 800, or 900 real samples.",
        "",
        "| Run | Status | Accuracy | Figure 3-6 Action | Output Root |",
        "| --- | --- | ---: | --- | --- |",
    ]
    if not candidates:
        lines.append("| none |  |  | no eligible 993 real-only/transfer result yet |  |")
    for row in candidates:
        run_name = row.get("run_name", "")
        if row.get("observed_accuracy") is None or row.get("status") == "pending":
            action = "wait for completed checker outputs"
        elif row.get("artifacts_usable") is False:
            action = "blocked until artifacts are usable"
        else:
            action = "draw Figure 3-6 diagnostics"
        accuracy = row.get("observed_accuracy")
        lines.append(
            "| {run} | `{status}` | {accuracy} | {action} | `{output}` |".format(
                run=run_name or Path(row.get("config", "")).stem,
                status=row.get("status"),
                accuracy="" if accuracy is None else f"{accuracy:.4f}",
                action=action,
                output=f"outputs/pre_gmm_figures/{run_name or Path(row.get('config', '')).stem}",
            )
        )
    if skipped:
        lines.extend(
            [
                "",
                f"- Confirmed skipped data-efficiency stems: {', '.join(sorted(set(skipped)))}.",
            ]
        )
    return lines


def hypothesis_lines(rows: list[dict]) -> list[str]:
    completed = [row for row in rows if row.get("observed_accuracy") is not None]
    failed = [row for row in rows if row.get("status") == "fail"]
    pending = [row for row in rows if row.get("status") == "pending"]
    out = []

    if pending:
        out.append(
            "- Many required runs are still pending; defer final ML hypotheses until verifier coverage is complete."
        )
    if failed:
        out.append(
            "- Prioritize failed runs whose methodology is valid but metrics miss target; these are optimization or data-fit failures, not reproduction-scope failures."
        )
    if completed:
        eces = []
        for row in completed:
            reliability = metrics_for(row, "reliability_plots")
            if "ece" in reliability:
                eces.append(float(reliability["ece"]))
        if eces:
            out.append(
                f"- Mean ECE across completed runs is {mean(eces):.4f}; high ECE would support calibration or confidence-sharpness diagnostics before architecture changes."
            )
    out.extend(
        [
            "- Compare real-only versus transfer per sample count first; if transfer gain shrinks at larger sample counts, inspect synthetic-real domain mismatch rather than increasing model capacity.",
            "- During long sweeps, run checker plus analyzer between experiment families and at least once during each large family; if loss is still steep or best validation occurs near the final epoch, extend the horizon before concluding that the architecture or data scheme failed.",
            "- Use worst per-class confusion pairs to decide whether errors track neighboring RPM/viscosity buckets, class imbalance, or visually similar flow regimes.",
            "- Use attention maps only as diagnostic evidence, not as pass/fail evidence; if attention artifacts are absent, request a separate diagnostic rerun with attention enabled after reproduction validity is established.",
            '- Use W&B gradient traces from `wandb.watch(log="all")` if available; if gradients are absent, do not infer stability from loss curves alone.',
        ]
    )
    return out


def structured_hypothesis_lines(rows: list[dict]) -> list[str]:
    real993 = next((row for row in rows if Path(row["config"]).stem == "realonly_993"), {})
    transfer993 = next((row for row in rows if Path(row["config"]).stem == "transfer_993"), {})
    transfer993_lrhold_rows = [
        row
        for row in rows
        if "transfer_993_microbatch_lrhold" in Path(row["config"]).stem
        or str(row.get("run_name", "")).startswith("repro_transfer_993_microbatch_lrhold")
    ]
    transfer993_lrhold = max(
        transfer993_lrhold_rows,
        key=lambda row: (
            row.get("observed_accuracy") is not None,
            row.get("observed_accuracy") or -1.0,
            str(row.get("run_name", "")),
        ),
        default={},
    )
    real993_losses = parse_losses(ROOT / real993["log_path"]) if real993 else []
    failed_realonly = [
        row for row in rows if row.get("status") == "fail" and Path(row["config"]).stem.startswith("realonly_")
    ]
    pending_transfer = [
        row for row in rows if row.get("status") == "pending" and Path(row["config"]).stem.startswith("transfer_")
    ]
    completed = [row for row in rows if row.get("observed_accuracy") is not None]

    lines = []
    if real993_losses:
        last = real993_losses[-1]
        best = min(real993_losses, key=lambda item: item["val_loss"])
        real993_status = real993.get("status")
        real993_accuracy = real993.get("observed_accuracy")
        if real993_status == "pass":
            inference = (
                "- Inference or hypothesis: the microbatch `realonly_993` gate succeeded; "
                "the earlier failed plain 30-epoch real-only family is consistent with insufficient optimizer-update density after the batch-size change."
            )
            prediction = (
                "- Prediction: rerunning the remaining real-only sample counts with the accepted optimizer-microbatch policy "
                "should materially improve their validation losses and confusion structure relative to the failed 30-epoch baselines."
            )
            falsifier = (
                "- Falsifying artifact or run: repeated real-only sample-count retries using the accepted policy still miss targets "
                "with the same low-accuracy confusion pattern seen in the failed baselines."
            )
            allowed = "- Allowed next action: train and verify `transfer_993` with the same microbatch policy, then compare the 993 real-only and transfer results before launching data-efficiency curves."
            blocked = "- Blocked next action: treating enhancement variants as valid reproduction or changing pass/fail targets after seeing this result."
        elif real993_status == "pending":
            inference = "- Inference or hypothesis: this run is still in-progress, so current loss movement is only a mid-run diagnostic for the optimizer-microbatch route."
            prediction = "- Prediction: if the update-density hypothesis is right, the completed microbatch `realonly_993` run should produce usable confusion/reliability artifacts and accuracy materially closer to the 0.723 reference than the failed plain 30-epoch baseline."
            falsifier = "- Falsifying artifact or run: completed microbatch `realonly_993` checker row with valid artifacts and accuracy still far below target, especially if validation loss has plateaued before the schedule end."
            allowed = "- Allowed next action: keep the existing tmux run alive and rerun checker plus analyzer after completion or a meaningful milestone."
            blocked = "- Blocked next action: launching the full data-efficiency curve or redefining the pass/fail standard before the 993 microbatch gate is verified."
        else:
            miss = None
            if real993_accuracy is not None and real993.get("target_accuracy") is not None:
                miss = real993.get("target_accuracy") - real993_accuracy
            if miss is not None and miss <= 0.01:
                inference = (
                    "- Inference or hypothesis: the microbatch real-only gate is a near miss rather than a broad reproduction collapse; "
                    "the final-epoch best validation loss suggests the 30-epoch microbatch horizon may still be slightly short for real-only training."
                )
                prediction = "- Prediction: a conservative same-method retry with a predeclared small horizon or scheduler adjustment should close the sub-1 percentage-point gap if optimization horizon is the limiting factor."
                falsifier = "- Falsifying artifact or run: repeated same-method 993 real-only retries remain below target with the same class 1/2/3 confusion pattern despite comparable or better validation loss."
                allowed = "- Allowed next action: compare against the completed 993 transfer run, inspect class confusions and provenance, then decide whether a bounded real-only gate retry is justified before full curves."
            else:
                inference = "- Inference or hypothesis: the microbatch gate did not validate the reproduction path; the failure should be treated as evidence for a data, preprocessing, split, checkpoint, or model-wiring mismatch before transfer runs."
                prediction = "- Prediction: artifact inspection should reveal persistent class-confusion structure or provenance mismatch explaining why longer training did not reach the reference target."
                falsifier = "- Falsifying artifact or run: a corrected provenance or wiring check followed by a rerun reaches the target without enhancement variants."
                allowed = "- Allowed next action: inspect confusion/reliability outputs, W&B config, dataset manifests, label mapping, and checkpoint provenance before launching another family."
            blocked = "- Blocked next action: launching the full data-efficiency curve or changing the reproduction standard while the real-only gate is failed."
        lines.extend(
            [
                "### Microbatch 993 real-only gate",
                "",
                f"- Evidence observed: `realonly_993` is `{real993_status}` at epoch {last['epoch']}/{last['total_epochs']}; observed accuracy is {real993_accuracy}; best validation loss is {best['val_loss']:.4f} at epoch {best['epoch']}; target accuracy is {real993.get('target_accuracy')}.",
                inference,
                prediction,
                falsifier,
                allowed,
                blocked,
                "",
            ]
        )
    elif real993:
        lines.extend(
            [
                "### Microbatch 993 real-only gate",
                "",
                f"- Evidence observed: `realonly_993` checker status is `{real993.get('status')}`; observed accuracy is {real993.get('observed_accuracy')}; target accuracy is {real993.get('target_accuracy')}.",
                "- Inference or hypothesis: the optimizer-microbatch gate has not yet produced complete local loss and artifact evidence.",
                "- Prediction: if update density explains the earlier failures, the microbatch gate should improve accuracy and confusion structure without requiring a 300-epoch route.",
                "- Falsifying artifact or run: completed microbatch `realonly_993` artifacts still miss the target with persistent collapse despite valid methodology.",
                "- Allowed next action: run or preserve the `realonly_993` microbatch gate and rerun checker plus analyzer after completion.",
                "- Blocked next action: launching full data-efficiency curves before the 993 real-only and transfer comparison exists.",
                "",
            ]
        )

    if failed_realonly:
        best_failed = max(failed_realonly, key=lambda row: row.get("observed_accuracy") or 0.0)
        lines.extend(
            [
                "### Below-target real-only gate",
                "",
                f"- Evidence observed: {len(failed_realonly)} completed real-only microbatch run is below target; best below-target run is `{best_failed['run_name']}` with accuracy {best_failed.get('observed_accuracy')} against target {best_failed.get('target_accuracy')}.",
                "- Inference or hypothesis: the current gate remains below the checker threshold even though it is much closer to the reference than the failed plain 30-epoch family.",
                "- Prediction: an optimizer-microbatch retry should improve validation loss and confusion structure before the same sample-count curve is relaunched.",
                "- Falsifying artifact or run: completed microbatch `realonly_993` and `transfer_993` gates with no accuracy or confusion-matrix improvement over the failed baseline.",
                "- Allowed next action: compare the completed 993 real-only and transfer confusion matrices against the paper/reference behavior before scheduling the full data-efficiency curve.",
                "- Blocked next action: treating the failed 30-epoch metrics as valid reproduction or using them to justify transfer-family conclusions.",
                "",
            ]
        )

    if pending_transfer:
        real993_gate = real993.get("status")
        transfer993_gate = transfer993.get("status")
        if real993_gate == "pass" and transfer993_gate == "pass":
            transfer_evidence = "- Evidence observed: both 993 gates are verified as `pass`; remaining transfer/data-efficiency configs are still pending."
            transfer_allowed = "- Allowed next action: compare 993 real-only versus 993 transfer gain, then launch the full microbatch data-efficiency curve if the comparison supports it."
            transfer_blocked = "- Blocked next action: treating unrun transfer/data-efficiency outputs as accepted before their checker rows pass."
        elif transfer993_gate == "pass" and real993_gate == "fail":
            transfer_evidence = "- Evidence observed: the 993 pair has completed with `transfer_993` passing and `realonly_993` failing just below target."
            transfer_allowed = "- Allowed next action: use the completed pair for diagnosis, but keep the full data-efficiency curve blocked until the real-only gate is accepted or the checklist is explicitly revised."
            transfer_blocked = "- Blocked next action: launching the full transfer/data-efficiency curve solely because transfer passed while the real-only gate remains failed."
        elif real993_gate == "pass":
            transfer_evidence = f"- Evidence observed: `realonly_993` is verified as `pass`, `transfer_993` is `{transfer993_gate}`, and {len(pending_transfer)} transfer configs are pending."
            transfer_allowed = "- Allowed next action: run `transfer_993` with the same optimizer-microbatch policy, then compare against the 993 real-only result."
            transfer_blocked = "- Blocked next action: launching the full transfer/data-efficiency curve before the 993 transfer comparison exists."
        else:
            transfer_evidence = f"- Evidence observed: {len(pending_transfer)} transfer configs are still pending, and the active 993 microbatch pair has not produced a verified comparison yet."
            transfer_allowed = "- Allowed next action: run or preserve the 993 microbatch gate pair in order, `realonly_993` then `transfer_993`, and compare their artifacts after checker/analyzer complete."
            transfer_blocked = "- Blocked next action: launching duplicate jobs or the full transfer/data-efficiency curve before the 993 pair comparison exists."
        lines.extend(
            [
                "### Transfer-family gating",
                "",
                transfer_evidence,
                "- Inference or hypothesis: transfer gains cannot be interpreted until the real-only baseline at the matching sample count is valid or explicitly marked failed by the checker.",
                "- Prediction: valid transfer runs should meet their own targets and show non-negative gain against verified real-only runs at the same sample count.",
                "- Falsifying artifact or run: checker rows where transfer accuracy is below the corresponding verified real-only accuracy or below the transfer reference target.",
                transfer_allowed,
                transfer_blocked,
                "",
            ]
        )

    if transfer993_lrhold:
        lrhold_losses = parse_losses(ROOT / transfer993_lrhold["log_path"])
        target = transfer993_lrhold.get("target_accuracy")
        observed = transfer993_lrhold.get("observed_accuracy")
        if lrhold_losses:
            best = min(lrhold_losses, key=lambda item: item["val_loss"])
            last = lrhold_losses[-1]
            loss_evidence = (
                f"best validation loss is {best['val_loss']:.4f} at epoch {best['epoch']}; "
                f"last epoch is {last['epoch']}/{last['total_epochs']} with validation loss {last['val_loss']:.4f}"
            )
        else:
            loss_evidence = "loss history is missing or pending"
        if transfer993_lrhold.get("status") == "pass":
            inference = (
                "- Inference or hypothesis: holding the LR before late cosine decay improved the synthetic-pretrained microbatch transfer route enough "
                "to beat the 300-epoch real-only diagnostic threshold."
            )
            allowed = "- Allowed next action: treat this transfer retry as accepted for the user-requested threshold, while keeping any separate real-only gate decision explicit."
            blocked = "- Blocked next action: treating the accuracy gain as calibrated confidence or as proof that every data-efficiency sample count will pass."
        else:
            inference = "- Inference or hypothesis: the LR-hold retry has not yet beaten the 300-epoch real-only diagnostic threshold."
            allowed = "- Allowed next action: inspect loss curve, class confusions, and W&B config before deciding another retry."
            blocked = "- Blocked next action: counting this retry as accepted before it exceeds the explicit threshold."
        lines.extend(
            [
                "### Transfer 993 LR-hold retry",
                "",
                f"- Evidence observed: `{transfer993_lrhold.get('run_name')}` is `{transfer993_lrhold.get('status')}` with accuracy {observed} against target {target}; {loss_evidence}.",
                inference,
                "- Prediction: if the improvement is from useful-LR dwell time, the class 1/2 confusion should improve relative to the 30-epoch microbatch transfer run.",
                "- Falsifying artifact or run: a repeated LR-hold transfer run fails to improve the same class confusions or only improves by test-set checkpoint selection noise.",
                allowed,
                blocked,
                "",
            ]
        )

    if completed:
        eces = []
        for row in completed:
            reliability = metrics_for(row, "reliability_plots")
            if "ece" in reliability:
                eces.append(float(reliability["ece"]))
        if eces:
            lines.extend(
                [
                    "### Calibration follow-up",
                    "",
                    f"- Evidence observed: completed usable runs with reliability metrics have mean ECE {mean(eces):.4f}.",
                    "- Inference or hypothesis: calibration is a diagnostic axis, not reproduction validity by itself; high ECE would support checking confidence sharpness after metric validity is known.",
                    "- Prediction: if calibration is a bottleneck, accuracy may improve less than confidence quality after calibration-only changes.",
                    "- Falsifying artifact or run: reliability metrics near reference quality while accuracy remains low.",
                    "- Allowed next action: inspect reliability plots after checker-valid runs exist.",
                    "- Blocked next action: counting calibration-only changes as valid reproduction enhancements.",
                    "",
                ]
            )

    return lines


def main() -> None:
    load_env()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = read_json(CHECKER_TABLE)
    if not isinstance(rows, list):
        rows = []
    references = read_json(REFERENCE_PATH)
    if not isinstance(references, dict):
        references = {}
    reference_summaries = fetch_reference_summaries(references)

    lines = [
        "# Viscnet Rebuild Analyzer Report",
        "",
        "This report is generated from `scripts/analyze_rebuild_results.py` and is downstream of `checklist.md` plus the checker report.",
        "",
        "## Verifier Dependency",
        "",
        f"- Checker table present: `{CHECKER_TABLE.relative_to(ROOT)}` = {CHECKER_TABLE.exists()}",
        f"- Checker report present: `{CHECKER_REPORT.relative_to(ROOT)}` = {CHECKER_REPORT.exists()}",
        "- Treat every hypothesis below as invalid for final decisions until the verifier marks the relevant run as `pass`, `fail`, or `blocked` with evidence.",
        "",
        "## Data Efficiency View",
        "",
        *data_efficiency_rows(rows),
        "",
        "## Pre-GMM Figure 3-6 Trigger",
        "",
        *pre_gmm_figure_trigger_rows(rows),
        "",
        "## Run Diagnostics",
        "",
    ]

    for row in rows:
        confusion = metrics_for(row, "confusion_matrix")
        reliability = metrics_for(row, "reliability_plots")
        losses = parse_losses(ROOT / row["log_path"])
        lines.append(f"### `{row['run_name']}`")
        lines.append("")
        lines.append(f"- Status: `{row.get('status')}`; methodology: `{row.get('methodology_fit')}`")
        lines.append(f"- Accuracy: observed={row.get('observed_accuracy')}, target={row.get('target_accuracy')}")
        if losses:
            last = losses[-1]
            best = min(losses, key=lambda item: item["val_loss"])
            lines.append(
                f"- Loss curve: last epoch {last['epoch']} train={last['train_loss']:.4f}, "
                f"val={last['val_loss']:.4f}; best val={best['val_loss']:.4f} at epoch {best['epoch']}"
            )
            lines.append(f"- Training horizon: {training_horizon_assessment(losses)}")
        if confusion:
            worst = ", ".join(f"class {c}: acc={a:.3f}, n={n}" for c, a, n in worst_classes(confusion))
            pairs = ", ".join(f"{t}->{p}: {n}" for t, p, n in largest_confusions(confusion))
            lines.append(f"- Worst classes: {worst or 'none'}")
            lines.append(f"- Largest confusions: {pairs or 'none'}")
        else:
            lines.append("- Confusion metrics: missing or pending.")
        if reliability:
            lines.append(f"- Calibration: ECE={reliability.get('ece')}, MCE={reliability.get('mce')}")
        else:
            lines.append("- Reliability metrics: missing or pending.")
        lines.append("")

    lines.extend(
        [
            "## Reference W&B Availability",
            "",
            f"- Reference summaries fetched: {len(reference_summaries)}",
        ]
    )
    for run_path, summary in sorted(reference_summaries.items()):
        if "error" in summary:
            lines.append(f"- `{run_path}`: unavailable ({summary['error']})")
        else:
            lines.append(f"- `{run_path}`: name=`{summary.get('name')}`, state=`{summary.get('state')}`")

    lines.extend(
        [
            "",
            "## Structured Hypotheses",
            "",
            *structured_hypothesis_lines(rows),
            "## Hypotheses To Test After Validity",
            "",
            *hypothesis_lines(rows),
            "",
        ]
    )
    ANALYZER_REPORT.write_text("\n".join(lines))
    print(f"Wrote {ANALYZER_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
