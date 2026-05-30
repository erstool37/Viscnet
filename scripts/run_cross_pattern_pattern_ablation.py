#!/usr/bin/env python3
"""Pattern-input ablation for completed cross-pattern checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets.VideoDatasetReal import VideoDatasetReal  # noqa: E402
from models.VivitEmbed import VivitEmbed  # noqa: E402
from utils.analysis import confusion_matrix  # noqa: E402

DEFAULT_CONFIGS = [
    "configs/rebuild/cross_pattern_1500_500/crosspat_train345_test1_no_rpm_lateconcat_lr5e5_gate001_ep70.yaml",
    "configs/rebuild/cross_pattern_1500_500/crosspat_train134_test5_no_rpm_lateconcat_lr5e5_gate001_ep70.yaml",
    "configs/rebuild/cross_pattern_1500_500/crosspat_train135_test4_no_rpm_lateconcat_lr5e5_gate001_ep70.yaml",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as file:
        return yaml.safe_load(file)


def build_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    cnn_cfg = model_cfg["cnn"]
    transformer_cfg = model_cfg["transformer"]
    embedding_cfg = model_cfg["embeddings"]
    frame_num = float(config["dataset"]["train"]["frame_num"])
    time = float(config["dataset"]["train"]["time"])
    model = VivitEmbed(
        float(cnn_cfg["drop_rate"]),
        int(cnn_cfg["output_size"]),
        bool(config["train_settings"]["classification"]),
        int(transformer_cfg["class"]),
        int(model_cfg["gmm"]["gmm_num"]),
        bool(embedding_cfg["rpm_bool"]),
        bool(embedding_cfg["pat_bool"]),
        num_frames=int(transformer_cfg.get("num_frames", int(frame_num * time))),
        image_size=int(transformer_cfg.get("image_size", 224)),
        pat_mode=str(embedding_cfg.get("pat_mode", "legacy")),
        pattern_gate_init=float(embedding_cfg.get("pattern_gate_init", 0.01)),
    ).to(device)
    raw_state = torch.load(checkpoint, map_location=device)
    state = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in raw_state.items()
    }
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def make_dataset(config: dict[str, Any]) -> VideoDatasetReal:
    section = config["dataset"]["test"]
    root = Path(section["test_root"])
    videos = sorted(str(path) for path in (root / config["misc_dir"]["video_subdir"]).glob("*.mp4"))
    params = sorted(str(path) for path in (root / config["misc_dir"]["norm_subdir"]).glob("*.json"))
    if not videos:
        raise FileNotFoundError(root / config["misc_dir"]["video_subdir"])
    if len(videos) != len(params):
        raise ValueError(f"Video/parameter count mismatch under {root}: {len(videos)} vs {len(params)}")
    return VideoDatasetReal(
        videos,
        params,
        float(section["frame_num"]),
        float(section["time"]),
        aug_bool=False,
        visc_class=int(config["model"]["transformer"]["class"]),
        temporal_window_config=section["dataloader"].get("temporal_window"),
    )


def load_forced_pattern(test_root: Path, pattern_id: int, device: torch.device) -> torch.Tensor:
    pattern_path = test_root / "backgrounds" / f"{int(pattern_id)}.png"
    image = cv2.imread(str(pattern_path))
    if image is None:
        raise FileNotFoundError(pattern_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    if height < 224 or width < 224:
        raise ValueError(f"Pattern image too small: {pattern_path} has {width}x{height}")
    top = (height - 224) // 2
    left = (width - 224) // 2
    image = image[top : top + 224, left : left + 224]
    image = (image / 127.5 - 1.0).astype(np.float32)
    return torch.tensor(image, dtype=torch.float32, device=device)


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    counts = np.asarray(metrics["confusion_matrix_counts"], dtype=float)
    labels = [int(label) for label in metrics.get("labels", list(range(counts.shape[1])))]
    predicted_counts = counts.sum(axis=0).astype(int).tolist()
    total = int(sum(predicted_counts))
    predicted_shares = [count / total for count in predicted_counts] if total else [0.0 for _ in predicted_counts]
    return {
        "accuracy": float(metrics["accuracy"]),
        "support": [int(value) for value in metrics["support"]],
        "per_class_accuracy": [float(value) for value in metrics["per_class_accuracy"]],
        "predicted_class_counts": {
            str(label): int(count) for label, count in zip(labels, predicted_counts)
        },
        "predicted_class_shares": {
            str(label): float(share) for label, share in zip(labels, predicted_shares)
        },
        "classes_used": int(sum(1 for count in predicted_counts if count > 0)),
        "max_predicted_class_share": float(max(predicted_shares) if predicted_shares else 0.0),
        "zero_predicted_classes": [label for label, count in zip(labels, predicted_counts) if count == 0],
    }


def pattern_for_mode(
    mode: str,
    batch_pattern: torch.Tensor,
    forced_patterns: dict[int, torch.Tensor],
) -> torch.Tensor:
    if mode == "correct":
        return batch_pattern
    if mode == "zero":
        return torch.zeros_like(batch_pattern)
    if mode.startswith("force"):
        pattern_id = int(mode.replace("force", ""))
        pattern = forced_patterns[pattern_id].unsqueeze(0).expand(batch_pattern.shape[0], -1, -1, -1)
        return pattern
    raise ValueError(f"Unsupported pattern ablation mode: {mode}")


def evaluate_mode(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    mode: str,
    forced_patterns: dict[int, torch.Tensor],
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    rpm_bool = bool(config["model"]["embeddings"]["rpm_bool"])
    criterion = torch.nn.CrossEntropyLoss()
    records: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.no_grad():
        for frames, _parameters, labels, names, rpm_idx, pattern in loader:
            frames = frames.to(device)
            labels = labels.to(device).view(-1).long()
            rpm_idx = rpm_idx.to(device).view(-1).long()
            if not rpm_bool:
                rpm_idx = torch.zeros_like(rpm_idx)
            pattern = pattern.to(device)
            pattern = pattern_for_mode(mode, pattern, forced_patterns)
            outputs = model(frames, rpm_idx, pattern)
            losses.append(float(criterion(outputs, labels).item()))
            preds = outputs.argmax(dim=1)
            for name, label, pred, logits_row in zip(names, labels, preds, outputs.detach().cpu().float()):
                records.append(
                    {
                        "name": str(name),
                        "target": int(label.detach().cpu().item()),
                        "prediction": int(pred.detach().cpu().item()),
                        "logits": [float(value) for value in logits_row.tolist()],
                    }
                )
    logits = np.asarray([record["logits"] for record in records], dtype=np.float32)
    labels = np.asarray([record["target"] for record in records], dtype=int)
    confusion_matrix(run_name, logits, labels, save_dir=str(output_dir))
    metrics_path = output_dir / f"{run_name}_metrics.json"
    with metrics_path.open("r") as file:
        metrics = json.load(file)
    summary = summarize_metrics(metrics)
    summary.update(
        {
            "mode": mode,
            "test_loss": float(np.mean(losses)) if losses else None,
            "metrics_path": str(metrics_path),
            "confusion_matrix_path": str(output_dir / f"{run_name}.png"),
        }
    )
    (output_dir / f"{run_name}_predictions.json").write_text(json.dumps(records, indent=2) + "\n")
    return summary


def evaluate_modes_single_pass(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    modes: list[str],
    forced_patterns: dict[int, torch.Tensor],
    output_dir: Path,
    run_name: str,
) -> list[dict[str, Any]]:
    rpm_bool = bool(config["model"]["embeddings"]["rpm_bool"])
    criterion = torch.nn.CrossEntropyLoss()
    records_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    losses_by_mode: dict[str, list[float]] = {mode: [] for mode in modes}
    with torch.no_grad():
        for frames, _parameters, labels, names, rpm_idx, pattern in loader:
            frames = frames.to(device)
            labels = labels.to(device).view(-1).long()
            rpm_idx = rpm_idx.to(device).view(-1).long()
            if not rpm_bool:
                rpm_idx = torch.zeros_like(rpm_idx)
            pattern = pattern.to(device)
            mode_patterns = [pattern_for_mode(mode, pattern, forced_patterns) for mode in modes]
            try:
                frames_all = torch.cat([frames] * len(modes), dim=0)
                rpm_all = torch.cat([rpm_idx] * len(modes), dim=0)
                pattern_all = torch.cat(mode_patterns, dim=0)
                outputs_all = model(frames_all, rpm_all, pattern_all)
                outputs_by_mode = list(outputs_all.split(frames.shape[0], dim=0))
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                outputs_by_mode = [model(frames, rpm_idx, mode_pattern) for mode_pattern in mode_patterns]

            for mode, outputs in zip(modes, outputs_by_mode):
                losses_by_mode[mode].append(float(criterion(outputs, labels).item()))
                preds = outputs.argmax(dim=1)
                for name, label, pred, logits_row in zip(names, labels, preds, outputs.detach().cpu().float()):
                    records_by_mode[mode].append(
                        {
                            "name": str(name),
                            "target": int(label.detach().cpu().item()),
                            "prediction": int(pred.detach().cpu().item()),
                            "logits": [float(value) for value in logits_row.tolist()],
                        }
                    )

    mode_results = []
    for mode in modes:
        mode_name = f"{run_name}_{mode}"
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        records = records_by_mode[mode]
        logits = np.asarray([record["logits"] for record in records], dtype=np.float32)
        labels = np.asarray([record["target"] for record in records], dtype=int)
        confusion_matrix(mode_name, logits, labels, save_dir=str(mode_dir))
        metrics_path = mode_dir / f"{mode_name}_metrics.json"
        with metrics_path.open("r") as file:
            metrics = json.load(file)
        summary = summarize_metrics(metrics)
        summary.update(
            {
                "mode": mode,
                "test_loss": float(np.mean(losses_by_mode[mode])) if losses_by_mode[mode] else None,
                "metrics_path": str(metrics_path),
                "confusion_matrix_path": str(mode_dir / f"{mode_name}.png"),
            }
        )
        (mode_dir / f"{mode_name}_predictions.json").write_text(json.dumps(records, indent=2) + "\n")
        mode_results.append(summary)
    return mode_results


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Cross-Pattern Pattern Ablation",
        "",
        "This diagnostic keeps the trained architecture and checkpoint fixed, then changes only the pattern image supplied at inference.",
        "",
        "| Run | Mode | Accuracy | Test loss | Classes used | Max pred share | Zero-pred classes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in summary["runs"]:
        for mode_result in run["modes"]:
            zeros = ",".join(str(value) for value in mode_result["zero_predicted_classes"])
            lines.append(
                f"| `{run['run_name']}` | `{mode_result['mode']}` | "
                f"{mode_result['accuracy']:.4f} | {mode_result['test_loss']:.4f} | "
                f"{mode_result['classes_used']} | {mode_result['max_predicted_class_share']:.3f} | "
                f"`[{zeros}]` |"
            )
    output_path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    modes = ["correct", "zero"] + [f"force{idx}" for idx in args.pattern_ids]
    all_results = {
        "diagnostic": "pattern_input_ablation",
        "note": "No architecture or weight changes; only inference-time pattern tensor changes.",
        "device": str(device),
        "modes": modes,
        "runs": [],
    }
    for config_arg in args.config:
        config_path = Path(config_arg)
        config = load_config(config_path)
        run_name = str(config["name"])
        checkpoint = Path(config["misc_dir"]["ckpt_root"]) / config["training"]["checkpoint_name"]
        test_root = Path(config["dataset"]["test"]["test_root"])
        model = build_model(config, checkpoint, device)
        dataset = make_dataset(config)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        forced_patterns = {
            pattern_id: load_forced_pattern(test_root, pattern_id, device) for pattern_id in args.pattern_ids
        }
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_results = {
            "run_name": run_name,
            "config": str(config_path),
            "checkpoint": str(checkpoint),
            "test_root": str(test_root),
            "sample_count": len(dataset),
            "rpm_bool": bool(config["model"]["embeddings"]["rpm_bool"]),
            "pat_mode": str(config["model"]["embeddings"].get("pat_mode", "legacy")),
            "modes": [],
        }
        print(f"Evaluating {run_name} modes={','.join(modes)} samples={len(dataset)}", flush=True)
        run_results["modes"] = evaluate_modes_single_pass(
            model=model,
            loader=loader,
            config=config,
            device=device,
            modes=modes,
            forced_patterns=forced_patterns,
            output_dir=run_dir,
            run_name=run_name,
        )
        all_results["runs"].append(run_results)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2) + "\n")
    write_markdown(all_results, output_root / "summary.md")
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--pattern-ids", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--output-root",
        default="outputs/rebuild_reproduction/cross_pattern_pattern_ablation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
