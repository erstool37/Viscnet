#!/usr/bin/env python3
"""Evaluate trained real-video models on modulo-frame reconstructions of real test videos."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.VivitEmbed import VivitEmbed  # noqa: E402
from utils.analysis import confusion_matrix  # noqa: E402


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_offsets(value: str, stride: int) -> list[int]:
    if value == "all":
        return list(range(stride))
    offsets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not offsets:
        raise ValueError("--phase-offsets must provide at least one offset")
    invalid = [offset for offset in offsets if offset < 0 or offset >= stride]
    if invalid:
        raise ValueError(f"phase offsets must be in [0, {stride - 1}], got {invalid}")
    return offsets


def read_video(path: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames: list[np.ndarray] = []
    width = height = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        frames.append(frame)
    cap.release()
    return frames, fps, (width, height)


def preprocess(frames_bgr: list[np.ndarray], image_size: int) -> torch.Tensor:
    processed = []
    for frame_bgr in frames_bgr:
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        processed.append((frame / 127.5 - 1.0).astype(np.float32))
    return torch.from_numpy(np.stack(processed)).permute(0, 3, 1, 2)


def write_clip(path: Path, frames_bgr: list[np.ndarray], fps: float, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()


def load_label_and_rpm(path: Path, visc_class: int) -> tuple[int, int]:
    data = json.loads(path.read_text())
    class_num = 10 // visc_class
    return int(data["visc_index"]) // class_num, int(data["rpm_idx"])


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
        image_size=int(transformer_cfg.get("image_size", 224)),
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def summarize_predictions(records: list[dict], num_classes: int, output_dir: Path, name: str) -> dict:
    logits = torch.tensor([row["logits"] for row in records], dtype=torch.float32)
    labels = [int(row["target"]) for row in records]
    confusion_matrix(name, logits.numpy(), labels, save_dir=str(output_dir))
    metrics_path = output_dir / f"{name}_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["evaluated_sample_count"] = len(labels)
    metrics["class_count"] = num_classes
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def run(args: argparse.Namespace) -> None:
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text())
    run_name = args.name or Path(config["training"]["checkpoint_name"]).stem
    output_root = ROOT / args.output_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint = ROOT / (args.checkpoint or Path(config["misc_dir"]["ckpt_root"]) / config["training"]["checkpoint_name"])
    test_root = ROOT / args.test_root
    para_root = test_root / "parametersNorm"
    videos = sorted((test_root / "videos").glob("*.mp4"))
    if args.max_videos:
        videos = videos[: args.max_videos]
    if not videos:
        raise SystemExit(f"No test videos found under {test_root / 'videos'}")

    transformer_cfg = config["model"]["transformer"]
    window_size = int(args.window_size or transformer_cfg.get("num_frames", 10))
    image_size = int(args.image_size or transformer_cfg.get("image_size", 224))
    num_classes = int(transformer_cfg["class"])
    offsets = parse_offsets(args.phase_offsets, args.sample_stride)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = build_model(config, checkpoint, device)

    source_records = []
    clip_records = []
    skipped = []
    clip_output = output_root / "clips" / "videos"
    with torch.no_grad():
        for video_index, video_path in enumerate(videos):
            para_path = para_root / f"{video_path.stem}.json"
            if not para_path.exists():
                raise FileNotFoundError(para_path)
            label, rpm_idx = load_label_and_rpm(para_path, num_classes)
            frames, fps, size = read_video(video_path)
            phase_logits = []
            for offset in offsets:
                indices = [offset + idx * args.sample_stride for idx in range(window_size)]
                if indices[-1] >= len(frames):
                    skipped.append(
                        {
                            "video": relative(video_path),
                            "phase_offset": offset,
                            "last_required_frame": indices[-1],
                            "source_frame_count": len(frames),
                        }
                    )
                    continue
                clip_frames = [frames[idx] for idx in indices]
                if args.materialize_clips:
                    clip_name = f"{video_path.stem}_mod_s{args.sample_stride}_p{offset}.mp4"
                    write_clip(clip_output / clip_name, clip_frames, fps / args.sample_stride, size)
                tensor = preprocess(clip_frames, image_size).unsqueeze(0).to(device)
                rpm = torch.tensor([rpm_idx], dtype=torch.long, device=device)
                pattern = torch.zeros((1, image_size, image_size, 3), dtype=torch.float32, device=device)
                logits = model(tensor, rpm, pattern).squeeze(0).detach().cpu()
                phase_logits.append(logits)
                clip_records.append(
                    {
                        "source_index": video_index,
                        "name": video_path.stem,
                        "phase_offset": offset,
                        "target": label,
                        "prediction": int(logits.argmax().item()),
                        "source_frame_indices": indices,
                        "logits": [float(value) for value in logits.tolist()],
                    }
                )
            if phase_logits:
                mean_logits = torch.stack(phase_logits).mean(dim=0)
                source_records.append(
                    {
                        "index": video_index,
                        "name": video_path.stem,
                        "target": label,
                        "prediction": int(mean_logits.argmax().item()),
                        "phase_count": len(phase_logits),
                        "logits": [float(value) for value in mean_logits.tolist()],
                    }
                )

    source_metrics = summarize_predictions(source_records, num_classes, output_root / "source_confusion", "source_mod_frame")
    clip_metrics = summarize_predictions(clip_records, num_classes, output_root / "clip_confusion", "clip_mod_frame")
    phase_counts: dict[int, dict[str, int]] = defaultdict(lambda: {"count": 0, "correct": 0})
    for row in clip_records:
        bucket = phase_counts[int(row["phase_offset"])]
        bucket["count"] += 1
        bucket["correct"] += int(row["target"] == row["prediction"])
    by_phase = {
        str(phase): {
            "count": values["count"],
            "accuracy": values["correct"] / values["count"] if values["count"] else None,
        }
        for phase, values in sorted(phase_counts.items())
    }

    summary = {
        "run_name": run_name,
        "config": args.config,
        "checkpoint": relative(checkpoint),
        "test_root": args.test_root,
        "source_count": len(source_records),
        "clip_count": len(clip_records),
        "window_size": window_size,
        "sample_stride": args.sample_stride,
        "phase_offsets": offsets,
        "source_accuracy": source_metrics["accuracy"],
        "clip_accuracy": clip_metrics["accuracy"],
        "by_phase": by_phase,
        "skipped_count": len(skipped),
        "skipped_preview": skipped[:25],
        "source_metrics": relative(output_root / "source_confusion" / "source_mod_frame_metrics.json"),
        "clip_metrics": relative(output_root / "clip_confusion" / "clip_mod_frame_metrics.json"),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_root / "source_predictions.json").write_text(json.dumps(source_records, indent=2) + "\n")
    (output_root / "clip_predictions.json").write_text(json.dumps(clip_records, indent=2) + "\n")
    probabilities = F.softmax(torch.tensor([row["logits"] for row in source_records]), dim=1)
    (output_root / "source_probabilities.json").write_text(
        json.dumps(
            [
                {"name": row["name"], "target": row["target"], "probabilities": [float(v) for v in probs]}
                for row, probs in zip(source_records, probabilities.tolist())
            ],
            indent=2,
        )
        + "\n"
    )
    print(
        f"{run_name}: source_accuracy={summary['source_accuracy']:.4f}; "
        f"clip_accuracy={summary['clip_accuracy']:.4f}; "
        f"sources={summary['source_count']}; clips={summary['clip_count']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name")
    parser.add_argument("--checkpoint")
    parser.add_argument("--test-root", default="dataset/RealArchive/test_1000_wo_pat2")
    parser.add_argument("--output-root", default="outputs/rebuild_reproduction/mod_frame_test_validation")
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--phase-offsets", default="all")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--materialize-clips", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
