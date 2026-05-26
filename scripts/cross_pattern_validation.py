#!/usr/bin/env python3
"""Evaluate real-video checkpoints on held-out render/pattern groups."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import confusion_matrix, load_weights, reliability_diagram  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def render_family(render: str) -> str:
    return re.sub(r"\d+$", "", render)


def build_records(dataset_root: Path, max_samples_per_render: int | None = None) -> list[dict]:
    records_by_render: dict[str, list[dict]] = defaultdict(list)
    norm_root = dataset_root / "parametersNorm"
    video_root = dataset_root / "videos"
    for para_path in sorted(norm_root.glob("*.json")):
        data = read_json(para_path)
        render = str(data.get("RENDER") or para_path.stem.split("_render")[-1])
        video_path = video_root / f"{para_path.stem}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        visc_index = int(data.get("visc_index", data.get("INDEX")))
        rpm_index = int(data.get("rpm_idx", data.get("RPM_index", data.get("rpm_index"))))
        records_by_render[render].append(
            {
                "name": para_path.stem,
                "render": render,
                "render_family": render_family(render),
                "visc_index": visc_index,
                "rpm_idx": rpm_index,
                "video_path": str(video_path),
                "parameters_norm_path": str(para_path),
            }
        )

    records = []
    for render in sorted(records_by_render):
        group = records_by_render[render]
        if max_samples_per_render is not None:
            group = group[:max_samples_per_render]
        records.extend(group)
    return records


def load_video(video_path: Path, frame_limit: int, image_size: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < frame_limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        frames.append((frame / 127.5 - 1.0).astype(np.float32))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    while len(frames) < frame_limit:
        frames.append(frames[-1].copy())
    return torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)


def one_hot(label: int, num_classes: int) -> np.ndarray:
    out = np.zeros(num_classes, dtype=np.float32)
    if 0 <= label < num_classes:
        out[label] = 1.0
    return out


def accuracy_for(records: list[dict], selector) -> float | None:
    subset = [row for row in records if selector(row)]
    if not subset:
        return None
    return sum(1 for row in subset if row["target"] == row["prediction"]) / len(subset)


def summarize_predictions(predictions: list[dict], num_classes: int) -> dict:
    targets = Counter(row["target"] for row in predictions)
    preds = Counter(row["prediction"] for row in predictions)
    correct = Counter(row["target"] for row in predictions if row["target"] == row["prediction"])
    by_render = {}
    by_family = {}
    for render in sorted({row["render"] for row in predictions}):
        subset = [row for row in predictions if row["render"] == render]
        by_render[render] = {
            "count": len(subset),
            "accuracy": accuracy_for(predictions, lambda row, render=render: row["render"] == render),
            "target_distribution": dict(sorted(Counter(row["target"] for row in subset).items())),
            "prediction_distribution": dict(sorted(Counter(row["prediction"] for row in subset).items())),
        }
    for family in sorted({row["render_family"] for row in predictions}):
        subset = [row for row in predictions if row["render_family"] == family]
        by_family[family] = {
            "count": len(subset),
            "accuracy": accuracy_for(predictions, lambda row, family=family: row["render_family"] == family),
            "target_distribution": dict(sorted(Counter(row["target"] for row in subset).items())),
            "prediction_distribution": dict(sorted(Counter(row["prediction"] for row in subset).items())),
        }
    per_class = {}
    for label in range(num_classes):
        support = targets[label]
        per_class[str(label)] = {
            "support": support,
            "correct": correct[label],
            "accuracy": correct[label] / support if support else None,
        }
    return {
        "sample_count": len(predictions),
        "accuracy": sum(1 for row in predictions if row["target"] == row["prediction"]) / len(predictions)
        if predictions
        else None,
        "target_distribution": dict(sorted(targets.items())),
        "prediction_distribution": dict(sorted(preds.items())),
        "per_class": per_class,
        "by_render": by_render,
        "by_render_family": by_family,
    }


def load_model(config: dict, checkpoint: Path, device: str):
    encoder_module = importlib.import_module(f"models.{config['model']['transformer']['encoder']}")
    encoder_class = getattr(encoder_module, config["model"]["transformer"]["encoder"])
    transformer_config = config["model"]["transformer"]
    model = encoder_class(
        float(config["model"]["cnn"]["drop_rate"]),
        int(config["model"]["cnn"]["output_size"]),
        bool(config["train_settings"]["classification"]),
        int(transformer_config["class"]),
        int(config["model"]["gmm"]["gmm_num"]),
        bool(config["model"]["embeddings"]["rpm_bool"]),
        bool(config["model"]["embeddings"]["pat_bool"]),
        num_frames=int(transformer_config.get("num_frames", 30)),
        image_size=int(transformer_config.get("image_size", 224)),
    ).to(device)
    load_weights(model, str(checkpoint))
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> dict:
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    run_name = args.name or Path(config["training"]["checkpoint_name"]).stem
    checkpoint = Path(args.checkpoint or Path(config["misc_dir"]["ckpt_root"]) / config["training"]["checkpoint_name"])
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root) / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    frame_limit = int(float(config["dataset"]["test"]["frame_num"]) * float(config["dataset"]["test"]["time"]))
    num_classes = int(config["model"]["transformer"]["class"])
    image_size = int(args.image_size or config["model"]["transformer"].get("image_size", 224))
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    records = build_records(dataset_root, args.max_samples_per_render)
    write_json(output_root / "manifest.json", records)

    model = load_model(config, checkpoint, device)
    predictions = []
    logits_for_plots = []
    targets_for_plots = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            frames = torch.stack(
                [load_video(Path(row["video_path"]), frame_limit, image_size) for row in batch]
            ).to(device)
            rpm_idx = torch.tensor([row["rpm_idx"] for row in batch], dtype=torch.long, device=device)
            pattern = torch.zeros((len(batch), image_size, image_size, 3), dtype=torch.float32, device=device)
            logits = model(frames, rpm_idx, pattern)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            logits_np = logits.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            for idx, row in enumerate(batch):
                target = int(row["visc_index"])
                predictions.append(
                    {
                        **row,
                        "target": target,
                        "prediction": int(pred_np[idx]),
                        "logits": logits_np[idx].astype(float).tolist(),
                        "probabilities": probs_np[idx].astype(float).tolist(),
                    }
                )
                logits_for_plots.append(logits_np[idx])
                targets_for_plots.append(one_hot(target, num_classes))

    summary = summarize_predictions(predictions, num_classes)
    summary.update(
        {
            "config": str(config_path),
            "checkpoint": str(checkpoint),
            "dataset_root": str(dataset_root),
            "frame_limit": frame_limit,
            "image_size": image_size,
            "max_samples_per_render": args.max_samples_per_render,
        }
    )
    write_json(output_root / "predictions.json", predictions)
    write_json(output_root / "metrics.json", summary)
    confusion_matrix(
        run_name,
        np.asarray(logits_for_plots),
        np.asarray([row["target"] for row in predictions]),
        class_names=[str(i) for i in range(num_classes)],
        save_dir=str(output_root / "confusion_matrix"),
    )
    reliability_diagram(logits_for_plots, targets_for_plots, name=run_name, save_dir=str(output_root / "reliability"))
    write_report(output_root / "report.md", run_name, summary)
    return summary


def write_report(path: Path, run_name: str, summary: dict) -> None:
    lines = [
        f"# Cross-Pattern Validation: {run_name}",
        "",
        f"- Accuracy: `{summary['accuracy']:.4f}`",
        f"- Samples: `{summary['sample_count']}`",
        f"- Dataset: `{summary['dataset_root']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        "",
        "## Render Families",
        "",
        "| Family | Count | Accuracy | Target Dist | Prediction Dist |",
        "|---|---:|---:|---|---|",
    ]
    for family, row in summary["by_render_family"].items():
        lines.append(
            f"| {family} | {row['count']} | {row['accuracy']:.4f} | "
            f"`{row['target_distribution']}` | `{row['prediction_distribution']}` |"
        )
    lines += ["", "## Renders", "", "| Render | Count | Accuracy |", "|---|---:|---:|"]
    for render, row in summary["by_render"].items():
        lines.append(f"| {render} | {row['count']} | {row['accuracy']:.4f} |")
    lines += ["", "## Per-Class Accuracy", "", "| Class | Support | Accuracy |", "|---:|---:|---:|"]
    for label, row in summary["per_class"].items():
        acc = "" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
        lines.append(f"| {label} | {row['support']} | {acc} |")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--name")
    parser.add_argument("--dataset-root", default="dataset/RealArchive/dualpatterndataset_V2_450")
    parser.add_argument("--output-root", default="outputs/rebuild_reproduction/cross_pattern_validation")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples-per-render", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    result = evaluate(parse_args())
    print(json.dumps({"accuracy": result["accuracy"], "sample_count": result["sample_count"]}, indent=2))
