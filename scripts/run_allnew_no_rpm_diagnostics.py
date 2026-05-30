#!/usr/bin/env python3
"""Post-hoc diagnostics for allnew no-RPM synthetic checkpoints.

This script does not train or update checkpoints. It can:
- run real-test inference with final-layer attention volumes,
- group attention summaries by true viscosity class and RPM metadata,
- draw a synthetic-validation confusion matrix with the same saved weight.

RPM metadata is still recorded for grouping, but model RPM input is forced to
zero whenever ``model.embeddings.rpm_bool`` is false.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import confusion_matrix  # noqa: E402

RPM_RE = re.compile(r"rpm(?P<rpm>\d+)")
VISC_RE = re.compile(r"visc(?P<visc>\d+(?:\.\d+)?)")


def init_distributed(device_arg: str) -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ:
        device = torch.device(device_arg if device_arg else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return 0, 1, device

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
    return rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(world_size: int, device: torch.device) -> None:
    if world_size <= 1:
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def gather_records(local_records: list[dict[str, Any]], world_size: int) -> list[dict[str, Any]]:
    if world_size == 1:
        return local_records
    bucket = [None] * world_size
    dist.all_gather_object(bucket, local_records)
    records: list[dict[str, Any]] = []
    for part in bucket:
        records.extend(part or [])
    return records


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return (index, *self.dataset[index])


def import_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def load_manifest_pairs(manifest_path: str) -> tuple[list[str], list[str]]:
    with open(manifest_path, "r") as file:
        payload = json.load(file)
    records = payload["samples"] if isinstance(payload, dict) else payload
    return [record["video_path"] for record in records], [record["parameters_norm_path"] for record in records]


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def resolve_checkpoint(config: dict[str, Any], checkpoint_arg: str | None) -> Path:
    ckpt_root = Path(config["misc_dir"]["ckpt_root"])
    checkpoint_name = checkpoint_arg or config["training"]["curr_ckpt"]
    checkpoint = Path(checkpoint_name)
    if not checkpoint.is_absolute():
        checkpoint = ckpt_root / checkpoint
    return checkpoint


def make_model(config: dict[str, Any], device: torch.device, output_attentions: bool) -> torch.nn.Module:
    encoder_name = config["model"]["transformer"]["encoder"]
    encoder_class = import_attr(f"models.{encoder_name}", encoder_name)
    frame_num = float(config["dataset"]["train"]["frame_num"])
    time = float(config["dataset"]["train"]["time"])
    model = encoder_class(
        float(config["model"]["cnn"]["drop_rate"]),
        int(config["model"]["cnn"]["output_size"]),
        bool(config["train_settings"]["classification"]),
        int(config["model"]["transformer"]["class"]),
        int(config["model"]["gmm"]["gmm_num"]),
        bool(config["model"]["embeddings"]["rpm_bool"]),
        bool(config["model"]["embeddings"]["pat_bool"]),
        num_frames=int(config["model"]["transformer"].get("num_frames", int(frame_num * time))),
        image_size=int(config["model"]["transformer"].get("image_size", 224)),
    ).to(device)
    if hasattr(model, "featureextractor") and hasattr(model.featureextractor, "config"):
        model.featureextractor.config.output_attentions = output_attentions
    return model


def load_state(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    raw_state = torch.load(checkpoint, map_location=device)
    if isinstance(raw_state, dict) and all(not torch.is_tensor(value) for value in raw_state.values()):
        for key in ("state_dict", "model", "weights"):
            if key in raw_state and isinstance(raw_state[key], dict):
                raw_state = raw_state[key]
                break
    state = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in raw_state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "loaded_keys": len(state),
    }


def split_dataset_paths(config: dict[str, Any], split: str) -> tuple[list[str], list[str], dict[str, Any], str]:
    video_subdir = config["misc_dir"]["video_subdir"]
    norm_subdir = config["misc_dir"]["norm_subdir"]

    if split == "real_test":
        section = config["dataset"]["test"]
        manifest = section.get("manifest")
        if manifest:
            video_paths, para_paths = load_manifest_pairs(manifest)
        else:
            root = Path(section["test_root"])
            video_paths = sorted(str(path) for path in (root / video_subdir).glob("*.mp4"))
            para_paths = sorted(str(path) for path in (root / norm_subdir).glob("*.json"))
        return video_paths, para_paths, section, section["dataloader"]["dataloader"]

    if split == "synthetic_val":
        section = config["dataset"]["train"]
        manifest = section.get("manifest")
        if manifest:
            video_paths, para_paths = load_manifest_pairs(manifest)
        else:
            root = Path(section["train_root"])
            video_paths = sorted(str(path) for path in (root / video_subdir).glob("*.mp4"))
            para_paths = sorted(str(path) for path in (root / norm_subdir).glob("*.json"))

        if bool(section.get("use_all_samples", False)):
            return video_paths[:1], para_paths[:1], section, section["dataloader"]["dataloader"]

        test_size = float(section["dataloader"]["test_size"])
        random_state = int(section["dataloader"]["random_state"])
        _, val_video_paths = train_test_split(video_paths, test_size=test_size, random_state=random_state)
        _, val_para_paths = train_test_split(para_paths, test_size=test_size, random_state=random_state)
        return val_video_paths, val_para_paths, section, section["dataloader"]["dataloader"]

    raise ValueError(f"Unsupported split: {split}")


def make_loader(
    config: dict[str, Any],
    split: str,
    batch_size: int,
    rank: int,
    world_size: int,
    max_samples: int | None,
) -> tuple[DataLoader, int]:
    video_paths, para_paths, section, dataset_name = split_dataset_paths(config, split)
    dataset_class = import_attr(f"datasets.{dataset_name}", dataset_name)
    dataset = dataset_class(
        video_paths,
        para_paths,
        float(section["frame_num"]),
        float(section["time"]),
        aug_bool=False,
        visc_class=int(config["model"]["transformer"]["class"]),
    )
    indexed = IndexedDataset(dataset)
    indices = list(range(len(indexed)))
    if max_samples is not None:
        indices = indices[:max_samples]
    local_indices = indices[rank::world_size]
    loader = DataLoader(Subset(indexed, local_indices), batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, len(indices)


def rpm_for_model(rpm_idx: torch.Tensor, rpm_bool: bool) -> torch.Tensor:
    rpm_idx = rpm_idx.long().view(-1)
    return rpm_idx if rpm_bool else torch.zeros_like(rpm_idx)


def tensor_to_int_list(values: torch.Tensor) -> list[int]:
    return [int(value) for value in values.detach().cpu().view(-1).tolist()]


def names_to_list(names: Any) -> list[str]:
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return [str(names)]


def parse_name_metadata(name: str) -> dict[str, float | int | None]:
    rpm_match = RPM_RE.search(name)
    visc_match = VISC_RE.search(name)
    return {
        "rpm_value": int(rpm_match.group("rpm")) if rpm_match else None,
        "viscosity_value": float(visc_match.group("visc")) if visc_match else None,
    }


def register_final_attention_hook(model: torch.nn.Module) -> tuple[list[Any], dict[str, torch.Tensor | None]]:
    captured: dict[str, torch.Tensor | None] = {"last": None}
    handles = []

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        candidate = output[-1] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(candidate) and candidate.ndim == 4 and candidate.shape[-1] == candidate.shape[-2]:
            captured["last"] = candidate.detach()

    for name, module in model.named_modules():
        lower = name.lower()
        if lower.endswith("attention.attention") or lower.endswith("selfattention"):
            handles.append(module.register_forward_hook(hook))
    if not handles:
        for name, module in model.named_modules():
            if "attention" in name.lower():
                handles.append(module.register_forward_hook(hook))
    return handles, captured


def attention_volume(attn: torch.Tensor, frames: torch.Tensor, sample_idx: int = 0, patch: int = 16) -> np.ndarray:
    _, _time_steps, _channels, height, width = frames.shape
    hp = height // patch
    wp = width // patch
    patches_per_frame = hp * wp
    attn_mean = attn[sample_idx].mean(dim=0)
    cls_to_tokens = attn_mean[0, 1:]
    groups = cls_to_tokens.numel() // patches_per_frame
    if groups <= 0:
        return np.zeros((hp, wp, 0), dtype=np.float32)
    cls_to_tokens = cls_to_tokens[: groups * patches_per_frame]
    volume = cls_to_tokens.reshape(groups, hp, wp).permute(1, 2, 0)
    return volume.float().cpu().numpy()


def entropy_norm(values: np.ndarray) -> float:
    flat = values.astype(np.float64).reshape(-1)
    total = float(np.nansum(flat))
    if total <= 0:
        return 0.0
    probs = flat / total
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / math.log(probs.size))


def normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)


def volume_stats(volume: np.ndarray) -> dict[str, float]:
    if volume.size == 0 or volume.shape[2] == 0:
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

    temporal = np.nansum(volume, axis=(0, 1))
    temporal_sum = float(np.nansum(temporal))
    temporal_probs = temporal / temporal_sum if temporal_sum > 0 else temporal
    thirds = np.array_split(np.arange(temporal.shape[0]), 3)
    early = float(np.nansum(temporal_probs[thirds[0]])) if len(thirds) > 0 else 0.0
    mid = float(np.nansum(temporal_probs[thirds[1]])) if len(thirds) > 1 else 0.0
    late = float(np.nansum(temporal_probs[thirds[2]])) if len(thirds) > 2 else 0.0

    spatial = np.nansum(volume, axis=2)
    spatial_sum = float(np.nansum(spatial))
    spatial_probs = spatial / spatial_sum if spatial_sum > 0 else spatial
    flat_spatial = spatial_probs.reshape(-1)
    top_k = max(1, int(math.ceil(0.10 * flat_spatial.size)))
    top10_mass = float(np.sort(flat_spatial)[-top_k:].sum())

    yy, xx = np.mgrid[0 : spatial.shape[0], 0 : spatial.shape[1]]
    if spatial_sum > 0:
        cy = float(np.nansum(yy * spatial_probs))
        cx = float(np.nansum(xx * spatial_probs))
        center_y = (spatial.shape[0] - 1) / 2.0
        center_x = (spatial.shape[1] - 1) / 2.0
        max_dist = math.hypot(center_y, center_x)
        center_distance = math.hypot(cy - center_y, cx - center_x) / max_dist if max_dist > 0 else 0.0
    else:
        center_distance = 0.0

    return {
        "temporal_entropy": entropy_norm(temporal),
        "spatial_entropy": entropy_norm(spatial),
        "temporal_peak_idx": float(np.nanargmax(temporal_probs)),
        "temporal_peak_frac": float(np.nanmax(temporal_probs)),
        "early_mass": early,
        "mid_mass": mid,
        "late_mass": late,
        "center_distance": center_distance,
        "top10_spatial_mass": top10_mass,
    }


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record.keys()}) if records else ["idx"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def group_summaries(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    summaries = []
    values = sorted({record[key] for record in records if record.get(key) is not None})
    metric_names = [
        "temporal_entropy",
        "spatial_entropy",
        "temporal_peak_idx",
        "temporal_peak_frac",
        "early_mass",
        "mid_mass",
        "late_mass",
        "center_distance",
        "top10_spatial_mass",
    ]
    for value in values:
        group = [record for record in records if record.get(key) == value]
        preds = [record["prediction"] for record in group]
        summaries.append(
            {
                "group_key": key,
                "group_value": value,
                "count": len(group),
                "accuracy": float(np.mean([record["correct"] for record in group])) if group else None,
                "prediction_counts": {str(label): int(preds.count(label)) for label in sorted(set(preds))},
                **{
                    f"{metric}_mean": float(np.mean([record[metric] for record in group])) if group else None
                    for metric in metric_names
                },
            }
        )
    return summaries


def load_group_mean_volume(records: list[dict[str, Any]], volume_dir: Path) -> np.ndarray | None:
    volumes = []
    for record in records:
        path = volume_dir / record["volume_file"]
        if path.exists():
            volume = np.load(path).astype(np.float32)
            if volume.ndim == 3 and volume.shape[2] > 0:
                volumes.append(volume)
    if not volumes:
        return None
    max_h = max(volume.shape[0] for volume in volumes)
    max_w = max(volume.shape[1] for volume in volumes)
    max_t = max(volume.shape[2] for volume in volumes)
    padded = np.full((len(volumes), max_h, max_w, max_t), np.nan, dtype=np.float32)
    for idx, volume in enumerate(volumes):
        padded[idx, : volume.shape[0], : volume.shape[1], : volume.shape[2]] = volume
    return np.nanmean(padded, axis=0)


def plot_group_panels(run_dir: Path, records: list[dict[str, Any]], volume_dir: Path, key: str, title: str) -> None:
    panel_dir = run_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    values = sorted({record[key] for record in records if record.get(key) is not None})
    if not values:
        return
    cols = len(values)
    fig, axes = plt.subplots(2, cols, figsize=(max(3.2 * cols, 4.0), 6.0), squeeze=False)
    for col, value in enumerate(values):
        group = [record for record in records if record.get(key) == value]
        mean_volume = load_group_mean_volume(group, volume_dir)
        ax_map = axes[0, col]
        ax_time = axes[1, col]
        if mean_volume is None:
            ax_map.axis("off")
            ax_time.axis("off")
            continue
        spatial = normalize(np.nanmean(normalize(mean_volume), axis=2))
        temporal = normalize(np.nanmean(normalize(mean_volume), axis=(0, 1)))
        accuracy = float(np.mean([record["correct"] for record in group])) if group else 0.0
        ax_map.imshow(spatial, cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
        ax_map.set_title(f"{value}\nn={len(group)} acc={accuracy:.2f}", fontsize=8)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_time.plot(np.arange(len(temporal)), temporal, marker="o", lw=1.2)
        ax_time.set_ylim(-0.05, 1.05)
        ax_time.set_xlabel("tubelet time")
        ax_time.set_ylabel("attention")
        ax_time.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(panel_dir / f"attention_by_{key}.png", dpi=180)
    plt.close(fig)


def save_group_summaries(run_dir: Path, records: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    summaries = []
    for key in group_keys:
        summaries.extend(group_summaries(records, key))
    with (run_dir / "attention_group_summary.json").open("w") as file:
        json.dump(summaries, file, indent=2)
    write_records_csv(run_dir / "attention_group_summary.csv", summaries)
    return summaries


def run_split_inference(
    *,
    config: dict[str, Any],
    model: torch.nn.Module,
    split: str,
    run_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    batch_size: int,
    max_samples: int | None,
    draw_confusion: bool,
) -> dict[str, Any] | None:
    loader, total_samples = make_loader(config, split, batch_size, rank, world_size, max_samples)
    rpm_bool = bool(config["model"]["embeddings"]["rpm_bool"])
    records = []
    model.eval()
    with torch.no_grad():
        for sample_indices, frames, parameters, hotvector, names, rpm_idx, pattern in loader:
            frames = frames.to(device)
            hotvector = hotvector.to(device)
            rpm_actual = rpm_idx.long().view(-1)
            rpm_model = rpm_for_model(rpm_idx.to(device), rpm_bool)
            pattern = pattern.to(device)
            logits = model(frames, rpm_model, pattern)
            preds = logits.argmax(dim=1).detach().cpu().view(-1).tolist()
            labels = hotvector.detach().cpu().view(-1).long().tolist()
            logits_list = logits.detach().cpu().float().tolist()
            kin_visc = parameters.detach().cpu()[:, 2].view(-1).tolist()
            name_list = names_to_list(names)
            for sample_idx, name, true, pred, rpm_value, kin_value, logits_row in zip(
                sample_indices.tolist(),
                name_list,
                labels,
                preds,
                tensor_to_int_list(rpm_actual),
                kin_visc,
                logits_list,
            ):
                name_meta = parse_name_metadata(name)
                records.append(
                    {
                        "idx": int(sample_idx),
                        "name": name,
                        "split": split,
                        "true_viscosity_class": int(true),
                        "prediction": int(pred),
                        "correct": int(pred) == int(true),
                        "rpm_idx": int(rpm_value),
                        "rpm_value": name_meta["rpm_value"],
                        "viscosity_value": name_meta["viscosity_value"],
                        "kinematic_viscosity_norm": float(kin_value),
                        "logits": [float(value) for value in logits_row],
                    }
                )

    records = sorted(gather_records(records, world_size), key=lambda item: item["idx"])
    distributed_barrier(world_size, device)
    if rank != 0:
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(run_dir / f"{split}_predictions.csv", records)
    with (run_dir / f"{split}_predictions.json").open("w") as file:
        json.dump(records, file, indent=2)
    summary = {
        "split": split,
        "count": len(records),
        "total_samples_selected": total_samples,
        "accuracy": float(np.mean([record["correct"] for record in records])) if records else None,
        "rpm_bool": rpm_bool,
        "rpm_input_policy": "actual_rpm_idx" if rpm_bool else "zero_tensor_for_model",
    }
    if draw_confusion and records:
        logits = np.asarray([record["logits"] for record in records], dtype=np.float32)
        labels = np.asarray([record["true_viscosity_class"] for record in records], dtype=int)
        confusion_matrix(
            f"{split}_confusion",
            logits,
            labels,
            save_dir=str(run_dir / "confusion_matrix"),
        )
        summary["confusion_metrics"] = str(run_dir / "confusion_matrix" / f"{split}_confusion_metrics.json")
    with (run_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    return summary


def run_real_attention(
    *,
    config: dict[str, Any],
    model: torch.nn.Module,
    run_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    batch_size: int,
    max_samples: int | None,
) -> dict[str, Any] | None:
    loader, total_samples = make_loader(config, "real_test", batch_size, rank, world_size, max_samples)
    volume_dir = run_dir / "volumes"
    if rank == 0:
        volume_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(world_size, device)

    rpm_bool = bool(config["model"]["embeddings"]["rpm_bool"])
    handles, captured = register_final_attention_hook(model)
    records = []
    model.eval()
    try:
        with torch.no_grad():
            for sample_indices, frames, parameters, hotvector, names, rpm_idx, pattern in loader:
                frames = frames.to(device)
                hotvector = hotvector.to(device)
                rpm_actual = rpm_idx.long().view(-1)
                rpm_model = rpm_for_model(rpm_idx.to(device), rpm_bool)
                pattern = pattern.to(device)
                captured["last"] = None
                logits = model(frames, rpm_model, pattern)
                attn = captured["last"]
                if attn is None:
                    raise RuntimeError("No final attention tensor was captured.")

                preds = logits.argmax(dim=1).detach().cpu().view(-1).tolist()
                labels = hotvector.detach().cpu().view(-1).long().tolist()
                logits_list = logits.detach().cpu().float().tolist()
                kin_visc = parameters.detach().cpu()[:, 2].view(-1).tolist()
                name_list = names_to_list(names)
                for batch_idx, (sample_idx, name, true, pred, rpm_value, kin_value, logits_row) in enumerate(
                    zip(
                        sample_indices.tolist(),
                        name_list,
                        labels,
                        preds,
                        tensor_to_int_list(rpm_actual),
                        kin_visc,
                        logits_list,
                    )
                ):
                    volume = attention_volume(attn, frames, sample_idx=batch_idx)
                    safe_name = name.replace("/", "_")
                    name_meta = parse_name_metadata(name)
                    rpm_group = name_meta["rpm_value"] if name_meta["rpm_value"] is not None else int(rpm_value)
                    visc_group = (
                        str(name_meta["viscosity_value"]).replace(".", "p")
                        if name_meta["viscosity_value"] is not None
                        else str(int(true))
                    )
                    volume_file = (
                        f"{int(sample_idx):05d}_{safe_name}_visc{visc_group}_rpm{rpm_group}_"
                        f"class{int(true)}_pred{int(pred)}_cls_attn_vol.npy"
                    )
                    np.save(volume_dir / volume_file, volume.astype(np.float32))
                    record = {
                        "idx": int(sample_idx),
                        "name": name,
                        "split": "real_test",
                        "true_viscosity_class": int(true),
                        "prediction": int(pred),
                        "correct": int(pred) == int(true),
                        "rpm_idx": int(rpm_value),
                        "rpm_value": name_meta["rpm_value"],
                        "viscosity_value": name_meta["viscosity_value"],
                        "kinematic_viscosity_norm": float(kin_value),
                        "volume_file": volume_file,
                        "logits": [float(value) for value in logits_row],
                    }
                    record.update(volume_stats(volume))
                    records.append(record)
    finally:
        for handle in handles:
            handle.remove()

    records = sorted(gather_records(records, world_size), key=lambda item: item["idx"])
    distributed_barrier(world_size, device)
    if rank != 0:
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(run_dir / "real_test_attention_metrics.csv", records)
    with (run_dir / "real_test_attention_metrics.json").open("w") as file:
        json.dump(records, file, indent=2)

    group_keys = ["viscosity_value", "rpm_value", "true_viscosity_class", "rpm_idx"]
    group_summary = save_group_summaries(run_dir, records, group_keys)
    plot_group_panels(run_dir, records, volume_dir, "viscosity_value", "Real Test Attention by Viscosity")
    plot_group_panels(run_dir, records, volume_dir, "rpm_value", "Real Test Attention by RPM")
    plot_group_panels(
        run_dir,
        records,
        volume_dir,
        "true_viscosity_class",
        "Real Test Attention by True Viscosity Class",
    )

    summary = {
        "split": "real_test",
        "count": len(records),
        "total_samples_selected": total_samples,
        "accuracy": float(np.mean([record["correct"] for record in records])) if records else None,
        "rpm_bool": rpm_bool,
        "rpm_input_policy": "actual_rpm_idx" if rpm_bool else "zero_tensor_for_model",
        "group_summary_count": len(group_summary),
        "volumes": str(volume_dir),
        "panels": str(run_dir / "panels"),
    }
    with (run_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Checkpoint filename or path. Defaults to training.curr_ckpt.")
    parser.add_argument("--output-root", default="outputs/rebuild_reproduction/allnew_no_rpm_diagnostics")
    parser.add_argument("--run-name", default="allnew_synth_no_rpm_augv1")
    parser.add_argument("--device", default="")
    parser.add_argument("--attention-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-real-samples", type=int, default=None)
    parser.add_argument("--max-synthetic-val-samples", type=int, default=None)
    parser.add_argument("--skip-real-attention", action="store_true")
    parser.add_argument("--skip-synthetic-val-confusion", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank, world_size, device = init_distributed(args.device)
    config = load_config(args.config)
    checkpoint = resolve_checkpoint(config, args.checkpoint)
    if rank == 0:
        print(f"Using checkpoint: {checkpoint}")
        print(f"RPM bool: {bool(config['model']['embeddings']['rpm_bool'])}")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    output_root = Path(args.output_root)
    run_dir = output_root / args.run_name
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(world_size, device)

    summaries = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "rpm_bool": bool(config["model"]["embeddings"]["rpm_bool"]),
        "rpm_input_policy": (
            "actual_rpm_idx" if bool(config["model"]["embeddings"]["rpm_bool"]) else "zero_tensor_for_model"
        ),
        "world_size": world_size,
    }

    model = make_model(config, device, output_attentions=not args.skip_real_attention)
    load_info = load_state(model, checkpoint, device)
    summaries["load_info"] = load_info

    if not args.skip_real_attention:
        real_summary = run_real_attention(
            config=config,
            model=model,
            run_dir=run_dir / "real_test_attention",
            rank=rank,
            world_size=world_size,
            device=device,
            batch_size=args.attention_batch_size,
            max_samples=args.max_real_samples,
        )
        if rank == 0:
            summaries["real_test_attention"] = real_summary

    if not args.skip_synthetic_val_confusion:
        synth_summary = run_split_inference(
            config=config,
            model=model,
            split="synthetic_val",
            run_dir=run_dir / "synthetic_val",
            rank=rank,
            world_size=world_size,
            device=device,
            batch_size=args.eval_batch_size,
            max_samples=args.max_synthetic_val_samples,
            draw_confusion=True,
        )
        if rank == 0:
            summaries["synthetic_val"] = synth_summary

    distributed_barrier(world_size, device)
    if rank == 0:
        with (run_dir / "summary.json").open("w") as file:
            json.dump(summaries, file, indent=2)
        print(f"Wrote {run_dir / 'summary.json'}")
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
