#!/usr/bin/env python3
"""Run 21-window test-time inference for the 30-frame real-only Viscnet model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.VivitEmbed import VivitEmbed  # noqa: E402
from utils.analysis import confusion_matrix  # noqa: E402


def init_distributed(device_arg: str) -> tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ:
        device = torch.device(device_arg if device_arg else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        return 0, 1, 0, device
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def gather_records(local_records: list[dict], world_size: int) -> list[dict]:
    if world_size == 1:
        return local_records
    bucket = [None] * world_size
    dist.all_gather_object(bucket, local_records)
    records: list[dict] = []
    for part in bucket:
        records.extend(part)
    return records


def load_rgb_frames(video_path: Path, required_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) < required_frames:
        raise ValueError(f"{video_path} has {len(frames)} frames; need at least {required_frames}")
    return frames[:required_frames]


def preprocess_window(frames: list[np.ndarray]) -> torch.Tensor:
    processed = []
    for frame in frames:
        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        processed.append((resized / 127.5 - 1.0).astype(np.float32))
    return torch.from_numpy(np.stack(processed)).permute(0, 3, 1, 2)


def load_label_and_rpm(path: Path, visc_class: int) -> tuple[int, int]:
    data = json.loads(path.read_text())
    class_num = 10 // visc_class
    label = int(data["visc_index"]) // class_num
    rpm_idx = int(data["rpm_idx"])
    return label, rpm_idx


def build_model(config: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    cnn_cfg = model_cfg["cnn"]
    transformer_cfg = model_cfg["transformer"]
    embedding_cfg = model_cfg["embeddings"]
    model = VivitEmbed(
        float(cnn_cfg["drop_rate"]),
        int(cnn_cfg["output_size"]),
        bool(config["train_settings"]["classification"]),
        int(transformer_cfg["class"]),
        int(model_cfg["gmm"]["gmm_num"]),
        bool(embedding_cfg["rpm_bool"]),
        bool(embedding_cfg["pat_bool"]),
        num_frames=int(transformer_cfg.get("num_frames", 30)),
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def top_confusions(cm_counts: list[list[int]], limit: int = 10) -> list[dict[str, int]]:
    pairs = []
    for true_idx, row in enumerate(cm_counts):
        for pred_idx, value in enumerate(row):
            if true_idx != pred_idx and value:
                pairs.append((int(value), true_idx, pred_idx))
    pairs.sort(reverse=True)
    return [{"true": true_idx, "pred": pred_idx, "count": count} for count, true_idx, pred_idx in pairs[:limit]]


def write_report(
    output_dir: Path,
    metrics: dict,
    baseline_metrics: dict,
    run_name: str,
    config_path: Path,
    checkpoint: Path,
) -> None:
    accuracy = float(metrics["accuracy"])
    baseline_accuracy = float(baseline_metrics["accuracy"])
    per_class = metrics["per_class_accuracy"]
    baseline_per_class = baseline_metrics["per_class_accuracy"]
    cm_counts = metrics["confusion_matrix_counts"]
    lines = [
        "# 21-Window Test-Time Inference",
        "",
        f"- Config: `{config_path.relative_to(ROOT)}`",
        f"- Checkpoint: `{checkpoint.relative_to(ROOT)}`",
        f"- Evaluated samples: `{sum(metrics['support'])}`",
        f"- Confusion-matrix total: `{sum(sum(row) for row in cm_counts)}`",
        f"- First-30-frame accuracy: `{baseline_accuracy:.4f}`",
        f"- 21-window averaged-logit accuracy: `{accuracy:.4f}`",
        f"- Accuracy delta: `{accuracy - baseline_accuracy:+.4f}`",
        "",
        "## Class 1/2/3",
        "",
        "| Class | First-30 | Window21 | Delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for class_idx in [1, 2, 3]:
        before = float(baseline_per_class[class_idx])
        after = float(per_class[class_idx])
        lines.append(f"| {class_idx} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
    lines.extend(["", "## Top Confusions", ""])
    for item in top_confusions(cm_counts, limit=12):
        lines.append(f"- `{item['true']} -> {item['pred']}`: {item['count']}")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "- Window averaging improves the first-30-frame baseline."
                if accuracy > baseline_accuracy
                else "- Window averaging does not improve the first-30-frame baseline."
            ),
        ]
    )
    (output_dir / f"{run_name}_report.md").write_text("\n".join(lines) + "\n")


def apply_config_defaults(config: dict, args: argparse.Namespace) -> argparse.Namespace:
    window_cfg = config.get("inference", {}).get("temporal_window", {})
    if window_cfg and not bool(window_cfg.get("enabled", False)):
        raise ValueError("Config inference.temporal_window.enabled is false.")
    if window_cfg and not bool(window_cfg.get("average_logits", True)):
        raise ValueError("window21_test_inference requires inference.temporal_window.average_logits=true.")

    args.full_frames = int(args.full_frames or window_cfg.get("full_frames", 50))
    args.window_size = int(args.window_size or window_cfg.get("window_size", 30))
    args.num_windows = int(args.num_windows or window_cfg.get("num_windows", 21))
    args.window_batch_size = int(args.window_batch_size or window_cfg.get("window_batch_size", 4))
    args.checkpoint = str(
        args.checkpoint
        or window_cfg.get("checkpoint")
        or Path(config["misc_dir"]["ckpt_root"]) / config["training"]["checkpoint_name"]
    )
    args.test_root = str(args.test_root or window_cfg.get("test_root") or config["dataset"]["test"]["test_root"])
    args.baseline_metrics = str(
        args.baseline_metrics
        or window_cfg.get("baseline_metrics")
        or Path(config["misc_dir"]["output_root"])
        / "confusion_matrix"
        / f"{Path(config['training']['checkpoint_name']).stem}_metrics.json"
    )
    args.output_dir = str(
        args.output_dir
        or window_cfg.get(
            "output_dir",
            "outputs/rebuild_reproduction/repro_realonly_993_window30x21_ep50/window21_test_inference",
        )
    )
    return args


def run(args: argparse.Namespace) -> None:
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text())
    args = apply_config_defaults(config, args)
    checkpoint = ROOT / args.checkpoint
    test_root = ROOT / args.test_root
    output_dir = ROOT / args.output_dir
    rank, world_size, _local_rank, device = init_distributed(args.device)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()

    visc_class = int(config["model"]["transformer"]["class"])
    model = build_model(config, checkpoint, device)

    videos = sorted((test_root / "videos").glob("*.mp4"))
    para_root = test_root / "parametersNorm"
    if not videos:
        raise SystemExit(f"No test videos found under {test_root / 'videos'}")
    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    local_records = []
    indexed_videos = list(enumerate(videos))
    pattern = torch.zeros((args.window_batch_size, 224, 224, 3), dtype=torch.float32, device=device)
    if rank == 0:
        print(
            f"window21 distributed inference: videos={len(videos)}; world_size={world_size}; "
            f"window_batch_size={args.window_batch_size}"
        )

    with torch.no_grad():
        for global_idx, video_path in indexed_videos[rank::world_size]:
            para_path = para_root / f"{video_path.stem}.json"
            if not para_path.exists():
                raise FileNotFoundError(para_path)
            label, rpm_idx = load_label_and_rpm(para_path, visc_class)
            frames = load_rgb_frames(video_path, args.full_frames)
            window_tensors = [
                preprocess_window(frames[start : start + args.window_size]) for start in range(args.num_windows)
            ]
            video_logits = []
            for start in range(0, len(window_tensors), args.window_batch_size):
                batch = torch.stack(window_tensors[start : start + args.window_batch_size]).to(device)
                rpm = torch.full((batch.shape[0],), rpm_idx, dtype=torch.long, device=device)
                pattern_batch = pattern[: batch.shape[0]]
                outputs = model(batch, rpm, pattern_batch)
                video_logits.append(outputs.detach().cpu())
            mean_logits = torch.cat(video_logits, dim=0).mean(dim=0)
            pred = int(mean_logits.argmax().item())
            local_records.append(
                {
                    "index": global_idx,
                    "name": video_path.stem,
                    "target": int(label),
                    "prediction": pred,
                    "logits": [float(value) for value in mean_logits.tolist()],
                }
            )

    records = sorted(gather_records(local_records, world_size), key=lambda item: item["index"])
    if rank != 0:
        cleanup_distributed(world_size)
        return
    logits = torch.tensor([record["logits"] for record in records], dtype=torch.float32)
    labels_all = [int(record["target"]) for record in records]
    run_name = "window21_test_inference"
    confusion_matrix(run_name, logits.numpy(), labels_all, save_dir=str(output_dir))

    metrics_path = output_dir / f"{run_name}_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.update(
        {
            "evaluated_sample_count": len(labels_all),
            "confusion_matrix_total": int(sum(sum(row) for row in metrics["confusion_matrix_counts"])),
            "num_windows_per_video": args.num_windows,
            "window_size": args.window_size,
            "full_frames_read": args.full_frames,
            "window_batch_size": args.window_batch_size,
            "averaging": "logits",
            "checkpoint": args.checkpoint,
            "config": args.config,
            "test_root": args.test_root,
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    probabilities = F.softmax(logits, dim=1)
    predictions_path = output_dir / f"{run_name}_predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "name": record["name"],
                    "target": int(record["target"]),
                    "prediction": int(record["prediction"]),
                    "logits": [float(v) for v in logit],
                    "probabilities": [float(v) for v in prob],
                }
                for record, logit, prob in zip(records, logits.tolist(), probabilities.tolist())
            ],
            indent=2,
        )
        + "\n"
    )

    baseline_path = ROOT / args.baseline_metrics
    baseline_metrics = json.loads(baseline_path.read_text())
    write_report(output_dir, metrics, baseline_metrics, run_name, config_path, checkpoint)
    print(
        f"window21 accuracy={metrics['accuracy']:.4f}; "
        f"samples={metrics['evaluated_sample_count']}; "
        f"cm_total={metrics['confusion_matrix_total']}"
    )
    cleanup_distributed(world_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rebuild/retries/realonly_993_window30x21_ep50.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test-root", default=None)
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--full-frames", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--num-windows", type=int, default=None)
    parser.add_argument("--window-batch-size", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
