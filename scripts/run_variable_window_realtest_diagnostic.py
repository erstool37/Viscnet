#!/usr/bin/env python3
"""Evaluate no-RPM real-test inference over variable 30-frame temporal windows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_allnew_no_rpm_diagnostics import (  # noqa: E402
    cleanup_distributed,
    distributed_barrier,
    gather_records,
    init_distributed,
    load_config,
    load_state,
    make_model,
    parse_name_metadata,
    resolve_checkpoint,
    rpm_for_model,
    split_dataset_paths,
)
from utils import confusion_matrix  # noqa: E402


def default_window_starts(window_size: int = 30, source_frame_count: int = 50) -> list[int]:
    if window_size <= 0 or source_frame_count < window_size:
        raise ValueError("source_frame_count must be >= window_size > 0")
    return list(range(source_frame_count - window_size + 1))


def _string_keyed(labels: list[int], values: list[Any], cast=float) -> dict[str, Any]:
    return {str(label): cast(value) for label, value in zip(labels, values)}


def summarize_confusion_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    counts = np.asarray(metrics["confusion_matrix_counts"], dtype=float)
    labels = [int(label) for label in metrics.get("labels", list(range(counts.shape[1])))]
    predicted_counts = counts.sum(axis=0).astype(int).tolist()
    total = int(sum(predicted_counts))
    if total:
        predicted_shares = [count / total for count in predicted_counts]
        max_share = max(predicted_shares)
    else:
        predicted_shares = [0.0 for _ in predicted_counts]
        max_share = 0.0
    zero_classes = [label for label, count in zip(labels, predicted_counts) if count == 0]
    return {
        "accuracy": metrics.get("accuracy"),
        "predicted_class_counts": _string_keyed(labels, predicted_counts, int),
        "predicted_class_shares": _string_keyed(labels, predicted_shares, float),
        "predicted_classes_used": int(sum(1 for count in predicted_counts if count > 0)),
        "max_predicted_class_share": float(max_share),
        "zero_predicted_classes": zero_classes,
    }


def infer_background_from_path(para_path: str) -> str:
    stem = Path(para_path).stem
    render = stem.split("_render")[-1]
    if render in set("ABCDEFGHIJ"):
        return "1"
    if render in set("KLMNO"):
        return "2"
    if render in set("PQRST"):
        return "3"
    if render in set("UVWXY"):
        return "4"
    return "1"


class VariableWindowRealDataset(Dataset):
    def __init__(
        self,
        video_paths: list[str],
        para_paths: list[str],
        starts: list[int],
        window_size: int,
        source_frame_count: int,
        visc_class: int,
    ) -> None:
        if len(video_paths) != len(para_paths):
            raise ValueError("video_paths and para_paths must have equal length")
        self.video_paths = video_paths
        self.para_paths = para_paths
        self.starts = [int(start) for start in starts]
        self.window_size = int(window_size)
        self.source_frame_count = int(source_frame_count)
        self.class_num = 10 // int(visc_class)
        self.cluster_map = {idx: idx // self.class_num for idx in range(10)}
        self.center_resize = A.Compose([A.Resize(224, 224, interpolation=1)])

    def __len__(self) -> int:
        return len(self.video_paths) * len(self.starts)

    def __getitem__(self, index: int):
        video_idx = int(index) // len(self.starts)
        start = self.starts[int(index) % len(self.starts)]
        video_path = self.video_paths[video_idx]
        para_path = self.para_paths[video_idx]
        parameters, label, pattern_name = self.load_parameters(para_path)
        frames = self.load_window(video_path, start)
        pattern = self.load_pattern(video_path, pattern_name)
        rpm = parameters[-1]
        return index, video_idx, start, frames, parameters, label, Path(para_path).stem, rpm, pattern

    def load_window(self, video_path: str, start: int) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.source_frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        end = start + self.window_size
        if len(frames) < end:
            raise ValueError(f"{video_path} has {len(frames)} frames; need at least {end}")
        window = frames[start:end]
        window = [self.center_resize(image=frame)["image"] for frame in window]
        window = [(frame / 127.5 - 1.0).astype(np.float32) for frame in window]
        return torch.tensor(np.stack(window)).permute(0, 3, 1, 2)

    def load_parameters(self, para_path: str):
        with Path(para_path).open("r") as file:
            data = json.load(file)
        density = data["density"]
        surf_t = data["surface_tension"]
        kin_visc = float(data["kinematic_viscosity"])
        rpm_idx = int(data["rpm_idx"])
        label = self.cluster_map[int(data["visc_index"])]
        pattern_name = str(data.get("background", infer_background_from_path(para_path)))
        return torch.tensor([density, surf_t, kin_visc, rpm_idx], dtype=torch.float32), torch.tensor(label), pattern_name

    def load_pattern(self, video_path: str, pattern_name: str) -> torch.Tensor:
        base_path = Path(video_path).parents[1]
        pattern_path = base_path / "backgrounds" / f"{pattern_name}.png"
        image = cv2.imread(str(pattern_path))
        if image is None:
            raise FileNotFoundError(pattern_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        top = (height - 224) // 2
        left = (width - 224) // 2
        image = image[top : top + 224, left : left + 224]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.tensor(image)


def make_loader(config: dict[str, Any], starts: list[int], args: argparse.Namespace, rank: int, world_size: int):
    video_paths, para_paths, section, _dataset_name = split_dataset_paths(config, "real_test")
    dataset = VariableWindowRealDataset(
        video_paths=video_paths,
        para_paths=para_paths,
        starts=starts,
        window_size=args.window_size,
        source_frame_count=args.source_frame_count,
        visc_class=int(config["model"]["transformer"]["class"]),
    )
    indices = list(range(len(dataset)))
    local_indices = indices[rank::world_size]
    loader = DataLoader(Subset(dataset, local_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    return loader, len(video_paths), len(dataset), section


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(records, file, indent=2)


def draw_and_summarize(name: str, records: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    logits = np.asarray([record["logits"] for record in records], dtype=np.float32)
    labels = np.asarray([record["true_viscosity_class"] for record in records], dtype=int)
    confusion_dir = run_dir / "confusion_matrix"
    confusion_matrix(name, logits, labels, save_dir=str(confusion_dir))
    metrics_path = confusion_dir / f"{name}_metrics.json"
    with metrics_path.open("r") as file:
        metrics = json.load(file)
    summary = summarize_confusion_metrics(metrics)
    summary["metrics_path"] = str(metrics_path)
    summary["confusion_matrix_path"] = str(confusion_dir / f"{name}.png")
    return summary


def aggregate_per_video(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["video_idx"])].append(record)
    aggregated = []
    for video_idx in sorted(grouped):
        group = sorted(grouped[video_idx], key=lambda item: int(item["window_start"]))
        logits = np.asarray([record["logits"] for record in group], dtype=np.float32)
        mean_logits = logits.mean(axis=0)
        first = group[0]
        pred = int(mean_logits.argmax())
        true = int(first["true_viscosity_class"])
        aggregated.append(
            {
                "video_idx": video_idx,
                "name": first["name"],
                "true_viscosity_class": true,
                "prediction": pred,
                "correct": pred == true,
                "window_starts": [int(record["window_start"]) for record in group],
                "logits": [float(value) for value in mean_logits.tolist()],
            }
        )
    return aggregated


def load_fixed_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    fixed_path = Path(path)
    if not fixed_path.exists():
        return {"metrics_path": str(fixed_path), "missing": True}
    with fixed_path.open("r") as file:
        metrics = json.load(file)
    summary = summarize_confusion_metrics(metrics)
    summary["metrics_path"] = str(fixed_path)
    summary["confusion_matrix_path"] = str(fixed_path.with_name(fixed_path.name.replace("_metrics.json", ".png")))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--source-frame-count", type=int, default=50)
    parser.add_argument("--fixed-first30-metrics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank, world_size, device = init_distributed("")
    config = load_config(args.config)
    starts = default_window_starts(args.window_size, args.source_frame_count)
    checkpoint = resolve_checkpoint(config, args.checkpoint)
    run_dir = Path(args.output_root) / args.run_name
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(world_size, device)

    rpm_bool = bool(config["model"]["embeddings"]["rpm_bool"])
    configured_num_frames = int(config["model"]["transformer"].get("num_frames", 0))
    if rpm_bool:
        raise ValueError("This diagnostic requires model.embeddings.rpm_bool: false")
    if configured_num_frames != args.window_size:
        raise ValueError(f"Expected model num_frames={args.window_size}, got {configured_num_frames}")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = make_model(config, device, output_attentions=False)
    load_info = load_state(model, checkpoint, device)
    loader, video_count, window_count, _section = make_loader(config, starts, args, rank, world_size)

    local_records = []
    loss_sum = 0.0
    loss_count = 0
    rpm_actual_nonzero = 0
    rpm_model_nonzero = 0
    model.eval()
    with torch.no_grad():
        for sample_idx, video_idx, start, frames, parameters, labels, names, rpm_idx, pattern in loader:
            frames = frames.to(device)
            labels = labels.to(device).long().view(-1)
            rpm_actual = rpm_idx.long().view(-1)
            rpm_model = rpm_for_model(rpm_idx.to(device), rpm_bool)
            pattern = pattern.to(device)
            logits = model(frames, rpm_model, pattern)
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").detach().cpu().item())
            loss_count += int(labels.numel())
            rpm_actual_nonzero += int((rpm_actual != 0).sum().item())
            rpm_model_nonzero += int((rpm_model.detach().cpu().view(-1) != 0).sum().item())
            preds = logits.argmax(dim=1).detach().cpu().tolist()
            logits_list = logits.detach().cpu().float().tolist()
            label_list = labels.detach().cpu().tolist()
            for idx, vid_idx, start_idx, name, true, pred, logits_row in zip(
                sample_idx.tolist(),
                video_idx.tolist(),
                start.tolist(),
                list(names),
                label_list,
                preds,
                logits_list,
            ):
                meta = parse_name_metadata(str(name))
                local_records.append(
                    {
                        "sample_idx": int(idx),
                        "video_idx": int(vid_idx),
                        "window_start": int(start_idx),
                        "name": str(name),
                        "true_viscosity_class": int(true),
                        "prediction": int(pred),
                        "correct": int(pred) == int(true),
                        "rpm_value": meta["rpm_value"],
                        "viscosity_value": meta["viscosity_value"],
                        "logits": [float(value) for value in logits_row],
                    }
                )

    gathered_payload = {
        "records": local_records,
        "loss_sum": loss_sum,
        "loss_count": loss_count,
        "rpm_actual_nonzero": rpm_actual_nonzero,
        "rpm_model_nonzero": rpm_model_nonzero,
    }
    gathered = gather_records([gathered_payload], world_size)
    distributed_barrier(world_size, device)
    if rank != 0:
        cleanup_distributed()
        return 0

    records = []
    total_loss_sum = 0.0
    total_loss_count = 0
    total_rpm_actual_nonzero = 0
    total_rpm_model_nonzero = 0
    for part in gathered:
        records.extend(part["records"])
        total_loss_sum += float(part["loss_sum"])
        total_loss_count += int(part["loss_count"])
        total_rpm_actual_nonzero += int(part["rpm_actual_nonzero"])
        total_rpm_model_nonzero += int(part["rpm_model_nonzero"])
    records = sorted(records, key=lambda item: int(item["sample_idx"]))
    per_video_records = aggregate_per_video(records)
    fixed_start0_records = [record for record in records if int(record["window_start"]) == 0]

    write_records(run_dir / "variable_window_predictions_per_window.json", records)
    write_records(run_dir / "variable_window_predictions_mean_logits_per_video.json", per_video_records)
    write_records(run_dir / "fixed_start0_predictions.json", fixed_start0_records)
    per_window_summary = draw_and_summarize("variable_window_all_starts_per_window", records, run_dir)
    per_video_summary = draw_and_summarize("variable_window_mean_logits_per_video", per_video_records, run_dir)
    fixed_start0_summary = draw_and_summarize("fixed_start0_window", fixed_start0_records, run_dir)
    fixed_summary = load_fixed_summary(args.fixed_first30_metrics)

    summary = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "run_name": args.run_name,
        "load_info": load_info,
        "rpm_bool": rpm_bool,
        "runtime_rpm_policy": "rpm tensor forced to zeros before model",
        "rpm_actual_nonzero_count": total_rpm_actual_nonzero,
        "rpm_model_nonzero_count": total_rpm_model_nonzero,
        "runtime_zeroed_rpm_confirmed": total_rpm_model_nonzero == 0,
        "configured_num_frames": configured_num_frames,
        "window_size": args.window_size,
        "source_frame_count": args.source_frame_count,
        "window_starts": starts,
        "video_count": video_count,
        "window_count": window_count,
        "test_loss_per_window": total_loss_sum / total_loss_count if total_loss_count else None,
        "fixed_start0_window": fixed_start0_summary,
        "variable_window_all_starts_per_window": per_window_summary,
        "variable_window_mean_logits_per_video": per_video_summary,
        "fixed_first30_reference": fixed_summary or fixed_start0_summary,
    }
    comparison_reference = fixed_summary if fixed_summary and not fixed_summary.get("missing") else fixed_start0_summary
    if comparison_reference and not comparison_reference.get("missing"):
        summary["distribution_delta_vs_fixed_first30"] = {
            "classes_used_delta_per_video": (
                per_video_summary["predicted_classes_used"] - comparison_reference["predicted_classes_used"]
            ),
            "max_predicted_class_share_delta_per_video": (
                per_video_summary["max_predicted_class_share"] - comparison_reference["max_predicted_class_share"]
            ),
            "zero_predicted_classes_delta_per_video": (
                len(per_video_summary["zero_predicted_classes"]) - len(comparison_reference["zero_predicted_classes"])
            ),
        }

    with (run_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
