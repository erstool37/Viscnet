#!/usr/bin/env python3
"""Grad-CAM diagnostic for the no-RPM 30-frame sliding-window Vivit weights."""

# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_allnew_no_rpm_diagnostics import load_state, make_model, resolve_checkpoint  # noqa: E402


DEFAULT_RUN_DIR = (
    "outputs/rebuild_reproduction/"
    "repro_realonly_993_window30x21_no_rpm_ep50_1000x21_inference_rerun/"
    "repro_realonly_993_window30x21_no_rpm_ep50_realtest_1000x21_windows_rerun"
)


def normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)


def entropy_norm(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(np.nansum(flat))
    if total <= 0:
        return 0.0
    probs = flat / total
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / math.log(probs.size))


def volume_stats(volume_hwt: np.ndarray) -> dict[str, float]:
    if volume_hwt.size == 0 or volume_hwt.shape[2] == 0:
        return {
            "temporal_entropy": 0.0,
            "spatial_entropy": 0.0,
            "temporal_peak_idx": 0.0,
            "temporal_peak_frac": 0.0,
            "early_mass": 0.0,
            "mid_mass": 0.0,
            "late_mass": 0.0,
            "center_distance": 0.0,
            "top10_spatial_mass": 0.0,
        }

    temporal = np.nansum(volume_hwt, axis=(0, 1))
    temporal_sum = float(np.nansum(temporal))
    temporal_probs = temporal / temporal_sum if temporal_sum > 0 else temporal
    groups = temporal.shape[0]
    thirds = np.array_split(np.arange(groups), 3)

    spatial = np.nansum(volume_hwt, axis=2)
    spatial_sum = float(np.nansum(spatial))
    spatial_probs = spatial / spatial_sum if spatial_sum > 0 else spatial
    flat_spatial = spatial_probs.reshape(-1)
    top_k = max(1, int(math.ceil(0.10 * flat_spatial.size)))

    yy, xx = np.mgrid[0 : spatial.shape[0], 0 : spatial.shape[1]]
    if spatial_sum > 0:
        cy = float(np.nansum(yy * spatial_probs))
        cx = float(np.nansum(xx * spatial_probs))
        center_y = (spatial.shape[0] - 1) / 2.0
        center_x = (spatial.shape[1] - 1) / 2.0
        max_dist = math.hypot(center_y, center_x)
        center_dist = math.hypot(cy - center_y, cx - center_x) / max_dist if max_dist > 0 else 0.0
    else:
        center_dist = 0.0

    return {
        "temporal_entropy": entropy_norm(temporal),
        "spatial_entropy": entropy_norm(spatial),
        "temporal_peak_idx": float(np.nanargmax(temporal_probs)) if temporal_probs.size else 0.0,
        "temporal_peak_frac": float(np.nanmax(temporal_probs)) if temporal_probs.size else 0.0,
        "early_mass": float(np.nansum(temporal_probs[thirds[0]])) if len(thirds) > 0 else 0.0,
        "mid_mass": float(np.nansum(temporal_probs[thirds[1]])) if len(thirds) > 1 else 0.0,
        "late_mass": float(np.nansum(temporal_probs[thirds[2]])) if len(thirds) > 2 else 0.0,
        "center_distance": center_dist,
        "top10_spatial_mass": float(np.sort(flat_spatial)[-top_k:].sum()) if flat_spatial.size else 0.0,
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as file:
        return yaml.safe_load(file)


def softmax_confidence(logits: list[float], label: int | None = None) -> float:
    probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=0)
    if label is None:
        return float(probs.max().item())
    return float(probs[int(label)].item())


def choose_video_records(
    per_video_records: list[dict[str, Any]],
    class_count: int,
    max_per_group: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for true_class in range(class_count):
        class_records = [r for r in per_video_records if int(r["true_viscosity_class"]) == true_class]
        for correct in [True, False]:
            group = [r for r in class_records if bool(r["correct"]) is correct]
            group = sorted(group, key=lambda r: softmax_confidence(r["logits"]), reverse=True)
            selected.extend(group[:max_per_group])
    return selected


def choose_window_record(video_record: dict[str, Any], per_window_records: list[dict[str, Any]]) -> dict[str, Any]:
    video_idx = int(video_record["video_idx"])
    target_pred = int(video_record["prediction"])
    group = [r for r in per_window_records if int(r["video_idx"]) == video_idx]
    if not group:
        raise ValueError(f"No per-window records found for video_idx={video_idx}")
    same_pred = [r for r in group if int(r["prediction"]) == target_pred]
    candidates = same_pred or group
    return max(candidates, key=lambda r: softmax_confidence(r["logits"], target_pred))


def split_dataset_paths(config: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    root = Path(config["dataset"]["test"]["test_root"])
    video_subdir = config["misc_dir"]["video_subdir"]
    norm_subdir = config["misc_dir"]["norm_subdir"]
    videos = {path.stem: str(path) for path in sorted((root / video_subdir).glob("*.mp4"))}
    params = {path.stem: str(path) for path in sorted((root / norm_subdir).glob("*.json"))}
    return videos, params


class WindowLoader:
    def __init__(self, source_frame_count: int, window_size: int, image_size: int = 224) -> None:
        self.source_frame_count = int(source_frame_count)
        self.window_size = int(window_size)
        self.image_size = int(image_size)
        self.resize = A.Compose([A.Resize(self.image_size, self.image_size, interpolation=1)])

    def load_window(self, video_path: str, start: int) -> tuple[torch.Tensor, np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.source_frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        end = int(start) + self.window_size
        if len(frames) < end:
            raise ValueError(f"{video_path} has {len(frames)} frames; need at least {end}")
        window = frames[int(start) : end]
        resized = [self.resize(image=frame)["image"] for frame in window]
        visible = np.stack(resized).astype(np.float32) / 255.0
        normalized = [(frame / 127.5 - 1.0).astype(np.float32) for frame in resized]
        tensor = torch.tensor(np.stack(normalized)).permute(0, 3, 1, 2).unsqueeze(0)
        return tensor, visible


class Conv3DGradCam:
    def __init__(self, model: torch.nn.Module, module_path: str) -> None:
        self.model = model
        self.module_path = module_path
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        module = self._resolve_module(module_path)
        self.handle_forward = module.register_forward_hook(self._forward_hook)
        self.handle_backward = module.register_full_backward_hook(self._backward_hook)

    def _resolve_module(self, module_path: str) -> torch.nn.Module:
        current: Any = self.model
        for part in module_path.split("."):
            current = getattr(current, part)
        if not isinstance(current, torch.nn.Module):
            raise TypeError(f"{module_path} is not a torch module")
        return current

    def _forward_hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        self.activation = output

    def _backward_hook(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradient = grad_output[0]

    def close(self) -> None:
        self.handle_forward.remove()
        self.handle_backward.remove()

    def volume(self) -> np.ndarray:
        volumes = self.volumes()
        return volumes[0]

    def volumes(self) -> np.ndarray:
        if self.activation is None or self.gradient is None:
            raise RuntimeError("Grad-CAM activation/gradient was not captured.")
        activation = self.activation.detach()
        gradient = self.gradient.detach()
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
        cams = torch.relu((weights * activation).sum(dim=1)).float().cpu().numpy()
        normalized = np.zeros_like(cams, dtype=np.float32)
        for index, cam in enumerate(cams):
            normalized[index] = normalize(cam)
        return np.transpose(normalized, (0, 2, 3, 1))


def render_overlay_panel(
    frames_rgb: np.ndarray,
    volume_hwt: np.ndarray,
    output_path: Path,
    title: str,
    tubelet_temporal_size: int,
) -> None:
    frame_indices = np.linspace(0, frames_rgb.shape[0] - 1, num=6).round().astype(int).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.0), squeeze=False)
    cmap = plt.get_cmap("jet")
    for ax, frame_idx in zip(axes.reshape(-1), frame_indices):
        tube_idx = min(volume_hwt.shape[2] - 1, max(0, int(frame_idx) // tubelet_temporal_size))
        heat = cv2.resize(volume_hwt[:, :, tube_idx], (frames_rgb.shape[2], frames_rgb.shape[1]))
        heat = normalize(heat)
        colored = cmap(heat)[..., :3]
        overlay = np.clip(0.55 * frames_rgb[frame_idx] + 0.45 * colored, 0.0, 1.0)
        ax.imshow(overlay)
        ax.set_title(f"frame {frame_idx} / tubelet {tube_idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def overlay_frame(
    frame_rgb: np.ndarray,
    heatmap_small: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    heat = cv2.resize(heatmap_small, (frame_rgb.shape[1], frame_rgb.shape[0]))
    heat = normalize(heat)
    colored = plt.get_cmap("jet")(heat)[..., :3]
    return np.clip((1.0 - alpha) * frame_rgb + alpha * colored, 0.0, 1.0)


def representative_frame_index(volume_hwt: np.ndarray, frame_count: int, tubelet_temporal_size: int) -> tuple[int, int]:
    temporal = np.nanmean(volume_hwt, axis=(0, 1))
    tube_idx = int(np.nanargmax(temporal)) if temporal.size else 0
    frame_idx = min(frame_count - 1, tube_idx * tubelet_temporal_size + tubelet_temporal_size // 2)
    return frame_idx, tube_idx


def select_example(records: list[dict[str, Any]], true_class: int, correct: bool) -> dict[str, Any] | None:
    group = [
        record
        for record in records
        if int(record["true_viscosity_class"]) == int(true_class) and bool(record["correct"]) is correct
    ]
    if not group:
        return None
    return max(group, key=lambda record: float(record.get("confidence", 0.0)))


def plot_class_example_overlay_grid(
    records: list[dict[str, Any]],
    volume_dir: Path,
    videos_by_name: dict[str, str],
    loader: WindowLoader,
    output_dir: Path,
    class_count: int,
    tubelet_temporal_size: int = 2,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gradcam_example_overlay_grid_correct_wrong.png"
    fig, axes = plt.subplots(2, class_count, figsize=(2.45 * class_count, 5.4), squeeze=False)

    for row_idx, (row_label, correct) in enumerate([("Correct", True), ("Wrong", False)]):
        for true_class in range(class_count):
            ax = axes[row_idx, true_class]
            record = select_example(records, true_class=true_class, correct=correct)
            if record is None:
                ax.axis("off")
                ax.set_title(f"class {true_class}\nno sample", fontsize=8)
                continue

            _, visible_frames = loader.load_window(videos_by_name[str(record["name"])], int(record["window_start"]))
            volume = np.load(volume_dir / record["volume_file"]).astype(np.float32)
            frame_idx, tube_idx = representative_frame_index(volume, visible_frames.shape[0], tubelet_temporal_size)
            tile = overlay_frame(visible_frames[frame_idx], volume[:, :, tube_idx])
            ax.imshow(tile)
            ax.set_title(
                "class {true}\npred {pred} conf {conf:.3f}\nw{window:02d} f{frame:02d} t{tube:02d}".format(
                    true=int(record["true_viscosity_class"]),
                    pred=int(record["prediction"]),
                    conf=float(record["confidence"]),
                    window=int(record["window_start"]),
                    frame=frame_idx,
                    tube=tube_idx,
                ),
                fontsize=7,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_idx, 0].set_ylabel(row_label, fontsize=11)

    fig.suptitle("Representative Grad-CAM Overlays by True Class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("w", newline="") as file:
        fieldnames = sorted({key for record in records for key in record.keys()})
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def mean_volume(records: list[dict[str, Any]], volume_dir: Path) -> np.ndarray | None:
    volumes = []
    for record in records:
        path = volume_dir / record["volume_file"]
        if path.exists():
            volumes.append(np.load(path).astype(np.float32))
    if not volumes:
        return None
    max_h = max(v.shape[0] for v in volumes)
    max_w = max(v.shape[1] for v in volumes)
    max_t = max(v.shape[2] for v in volumes)
    padded = np.full((len(volumes), max_h, max_w, max_t), np.nan, dtype=np.float32)
    for idx, volume in enumerate(volumes):
        padded[idx, : volume.shape[0], : volume.shape[1], : volume.shape[2]] = volume
    return np.nanmean(padded, axis=0)


def plot_group_panels(records: list[dict[str, Any]], volume_dir: Path, output_dir: Path, class_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, class_count, figsize=(2.2 * class_count, 5.0), squeeze=False)
    for row_idx, (row_name, correct) in enumerate([("Correct", True), ("Wrong", False)]):
        for true_class in range(class_count):
            ax = axes[row_idx, true_class]
            group = [
                record
                for record in records
                if int(record["true_viscosity_class"]) == true_class and bool(record["correct"]) is correct
            ]
            volume = mean_volume(group, volume_dir)
            if volume is None:
                ax.axis("off")
                ax.set_title(f"class {true_class}\nn=0", fontsize=8)
                continue
            spatial = normalize(np.nanmean(normalize(volume), axis=2))
            ax.imshow(spatial, cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
            pred_counts: dict[int, int] = {}
            for record in group:
                pred = int(record["prediction"])
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            pred_text = ",".join(f"{pred}:{count}" for pred, count in sorted(pred_counts.items()))
            ax.set_title(f"class {true_class}\nn={len(group)}\npred {pred_text}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_idx, 0].set_ylabel(row_name, fontsize=11)
    fig.suptitle("Grad-CAM spatial maps by true class and correctness")
    fig.tight_layout()
    fig.savefig(output_dir / "gradcam_by_true_class_correct_wrong_spatial.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), squeeze=False)
    for ax, title, correct in [(axes[0, 0], "Correct", True), (axes[0, 1], "Wrong", False)]:
        for true_class in range(class_count):
            group = [
                record
                for record in records
                if int(record["true_viscosity_class"]) == true_class and bool(record["correct"]) is correct
            ]
            volume = mean_volume(group, volume_dir)
            if volume is None:
                continue
            temporal = normalize(np.nanmean(normalize(volume), axis=(0, 1)))
            ax.plot(temporal, marker="o", lw=1.1, ms=2.5, label=f"class {true_class} n={len(group)}")
        ax.set_title(title)
        ax.set_xlabel("tubelet time")
        ax.set_ylabel("Grad-CAM")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Grad-CAM temporal curves by true class and correctness")
    fig.tight_layout()
    fig.savefig(output_dir / "gradcam_by_true_class_correct_wrong_temporal.png", dpi=220)
    plt.close(fig)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(records),
        "correct_count": sum(1 for r in records if r["correct"]),
        "wrong_count": sum(1 for r in records if not r["correct"]),
    }
    for prefix, group in {
        "all": records,
        "correct": [record for record in records if record["correct"]],
        "wrong": [record for record in records if not record["correct"]],
    }.items():
        summary[f"{prefix}_count"] = len(group)
        for metric in [
            "temporal_entropy",
            "spatial_entropy",
            "temporal_peak_idx",
            "temporal_peak_frac",
            "early_mass",
            "mid_mass",
            "late_mass",
            "center_distance",
            "top10_spatial_mass",
            "confidence",
        ]:
            values = [float(record[metric]) for record in group if record.get(metric) is not None]
            summary[f"{prefix}_{metric}_mean"] = float(np.mean(values)) if values else None
    return summary


def write_report(path: Path, summary: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# No-RPM Sliding-Window Grad-CAM Diagnostic",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Config: `{summary['config']}`",
        f"- Samples: {summary['sample_count']} ({summary['correct_count']} correct, {summary['wrong_count']} wrong)",
        f"- Runtime RPM tensor nonzero count: {summary['rpm_model_nonzero_count']}",
        f"- Target layer: `{summary['target_layer']}`",
        f"- Grad-CAM target: `{summary['gradcam_target']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| group | n | spatial entropy | temporal entropy | top10 spatial mass | early/mid/late mass | peak tubelet |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for group in ["all", "correct", "wrong"]:
        n = summary.get(f"{group}_count")
        lines.append(
            "| {group} | {n} | {se:.4f} | {te:.4f} | {top:.4f} | {early:.3f}/{mid:.3f}/{late:.3f} | {peak:.2f} |".format(
                group=group,
                n=n,
                se=summary.get(f"{group}_spatial_entropy_mean") or 0.0,
                te=summary.get(f"{group}_temporal_entropy_mean") or 0.0,
                top=summary.get(f"{group}_top10_spatial_mass_mean") or 0.0,
                early=summary.get(f"{group}_early_mass_mean") or 0.0,
                mid=summary.get(f"{group}_mid_mass_mean") or 0.0,
                late=summary.get(f"{group}_late_mass_mean") or 0.0,
                peak=summary.get(f"{group}_temporal_peak_idx_mean") or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Sample overlays: `{output_dir / 'overlays'}`",
            f"- Grad-CAM volumes: `{output_dir / 'volumes'}`",
            f"- Representative overlay grid: `{output_dir / 'panels/gradcam_example_overlay_grid_correct_wrong.png'}`",
            f"- Group spatial panel: `{output_dir / 'panels/gradcam_by_true_class_correct_wrong_spatial.png'}`",
            f"- Group temporal panel: `{output_dir / 'panels/gradcam_by_true_class_correct_wrong_temporal.png'}`",
            f"- Per-sample metrics: `{output_dir / 'sample_gradcam_metrics.csv'}`",
            "",
            "## Interpretation Notes",
            "",
            "- This is a bounded attribution diagnostic, not a full causal proof.",
            "- Grad-CAM is computed on the ViViT tubelet Conv3d projection, so maps are low-resolution tubelet saliency.",
            "- Higher top10 spatial mass means the model's gradient evidence is more spatially concentrated.",
            "- Early/mid/late mass shows which part of the 30-frame window most influenced the predicted class score.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rebuild/retries/realonly_993_window30x21_no_rpm_ep50.yaml")
    parser.add_argument("--checkpoint", default="repro_realonly_993_window30x21_no_rpm_ep50.pth")
    parser.add_argument("--prediction-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", default="outputs/rebuild_reproduction/gradcam_no_rpm_window30x21")
    parser.add_argument("--max-per-group", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--source-frame-count", type=int, default=50)
    parser.add_argument("--target-layer", default="featureextractor.embeddings.patch_embeddings.projection")
    parser.add_argument("--gradcam-target", choices=["prediction", "true"], default="prediction")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if bool(config["model"]["embeddings"]["rpm_bool"]):
        raise ValueError("This diagnostic requires model.embeddings.rpm_bool: false")
    configured_frames = int(config["model"]["transformer"].get("num_frames", 0))
    if configured_frames != int(args.window_size):
        raise ValueError(f"Expected model num_frames={args.window_size}, got {configured_frames}")

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint = resolve_checkpoint(config, args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    prediction_dir = Path(args.prediction_dir)
    per_video = load_json(prediction_dir / "variable_window_predictions_mean_logits_per_video.json")
    per_window = load_json(prediction_dir / "variable_window_predictions_per_window.json")
    class_count = int(config["model"]["transformer"]["class"])
    selected_videos = choose_video_records(per_video, class_count=class_count, max_per_group=args.max_per_group)
    selected_windows = [choose_window_record(video, per_window) for video in selected_videos]

    videos_by_name, _params_by_name = split_dataset_paths(config)
    loader = WindowLoader(source_frame_count=args.source_frame_count, window_size=args.window_size)
    output_dir = Path(args.output_dir)
    volume_dir = output_dir / "volumes"
    overlay_dir = output_dir / "overlays"
    panel_dir = output_dir / "panels"
    volume_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(config, device, output_attentions=False)
    load_info = load_state(model, checkpoint, device)
    model.eval()
    cam = Conv3DGradCam(model, args.target_layer)

    records: list[dict[str, Any]] = []
    rpm_model_nonzero = 0
    try:
        for index, record in enumerate(selected_windows):
            name = str(record["name"])
            if name not in videos_by_name:
                raise KeyError(f"Missing video path for {name}")
            frames, visible_frames = loader.load_window(videos_by_name[name], int(record["window_start"]))
            frames = frames.to(device)
            rpm_idx = torch.zeros((1,), dtype=torch.long, device=device)
            pattern = torch.zeros((1, 224, 224, 3), dtype=torch.float32, device=device)
            rpm_model_nonzero += int((rpm_idx.detach().cpu() != 0).sum().item())

            model.zero_grad(set_to_none=True)
            logits = model(frames, rpm_idx, pattern)
            prediction = int(logits.argmax(dim=1).item())
            true_class = int(record["true_viscosity_class"])
            target_class = prediction if args.gradcam_target == "prediction" else true_class
            score = logits[0, target_class]
            score.backward()
            volume = cam.volume()

            volume_file = (
                f"{index:03d}_{safe_name(name)}_w{int(record['window_start']):02d}_"
                f"class{true_class}_pred{prediction}_{args.gradcam_target}_gradcam.npy"
            )
            np.save(volume_dir / volume_file, volume.astype(np.float32))
            overlay_file = volume_file.replace(".npy", ".png")
            render_overlay_panel(
                visible_frames,
                volume,
                overlay_dir / overlay_file,
                title=f"{name} | start={record['window_start']} | true={true_class} pred={prediction}",
                tubelet_temporal_size=2,
            )

            probs = F.softmax(logits.detach().cpu().float()[0], dim=0)
            out = {
                "idx": index,
                "video_idx": int(record["video_idx"]),
                "sample_idx": int(record["sample_idx"]),
                "name": name,
                "window_start": int(record["window_start"]),
                "true_viscosity_class": true_class,
                "prediction": prediction,
                "correct": prediction == true_class,
                "rpm_value": record.get("rpm_value"),
                "viscosity_value": record.get("viscosity_value"),
                "target_class": int(target_class),
                "confidence": float(probs[prediction].item()),
                "volume_file": volume_file,
                "overlay_file": overlay_file,
            }
            out.update(volume_stats(volume))
            records.append(out)
    finally:
        cam.close()

    write_csv(output_dir / "sample_gradcam_metrics.csv", records)
    (output_dir / "sample_gradcam_metrics.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    plot_group_panels(records, volume_dir, panel_dir, class_count=class_count)
    example_grid_path = plot_class_example_overlay_grid(
        records,
        volume_dir,
        videos_by_name,
        loader,
        panel_dir,
        class_count=class_count,
    )
    summary = summarize(records)
    summary.update(
        {
            "config": str(config_path),
            "checkpoint": str(checkpoint),
            "load_info": load_info,
            "prediction_dir": str(prediction_dir),
            "output_dir": str(output_dir),
            "target_layer": args.target_layer,
            "gradcam_target": args.gradcam_target,
            "rpm_bool": False,
            "rpm_model_nonzero_count": rpm_model_nonzero,
            "configured_num_frames": configured_frames,
            "window_size": int(args.window_size),
            "source_frame_count": int(args.source_frame_count),
            "max_per_group": int(args.max_per_group),
            "representative_overlay_grid": str(example_grid_path),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "report.md", summary, output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
