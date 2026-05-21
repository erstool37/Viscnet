#!/usr/bin/env python3
"""Generate final-layer attention diagnostics for selected 993 checkpoints.

This is a test-only diagnostic. It does not train or modify checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


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
        records.extend(part)
    return records


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return (index, *self.dataset[index])


def load_manifest_pairs(manifest_path: str) -> tuple[list[str], list[str]]:
    with open(manifest_path, "r") as file:
        payload = json.load(file)
    records = payload["samples"] if isinstance(payload, dict) else payload
    video_paths = [record["video_path"] for record in records]
    para_paths = [record["parameters_norm_path"] for record in records]
    return video_paths, para_paths


def import_attr(module_name: str, attr_name: str) -> Any:
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)


def make_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    encoder_name = config["model"]["transformer"]["encoder"]
    encoder_class = import_attr(f"models.{encoder_name}", encoder_name)
    model = encoder_class(
        float(config["model"]["cnn"]["drop_rate"]),
        int(config["model"]["cnn"]["output_size"]),
        bool(config["train_settings"]["classification"]),
        int(config["model"]["transformer"]["class"]),
        int(config["model"]["gmm"]["gmm_num"]),
        bool(config["model"]["embeddings"]["rpm_bool"]),
        bool(config["model"]["embeddings"]["pat_bool"]),
        num_frames=int(
            config["model"]["transformer"].get(
                "num_frames",
                int(float(config["dataset"]["test"]["frame_num"]) * float(config["dataset"]["test"]["time"])),
            )
        ),
    ).to(device)
    if hasattr(model, "featureextractor") and hasattr(model.featureextractor, "config"):
        model.featureextractor.config.output_attentions = True
    return model


def make_test_loader(
    config: dict[str, Any], batch_size: int, rank: int, world_size: int, max_samples: int | None
) -> DataLoader:
    dataset_name = config["dataset"]["test"]["dataloader"]["dataloader"]
    dataset_class = import_attr(f"datasets.{dataset_name}", dataset_name)
    test_manifest = config["dataset"]["test"].get("manifest")
    if test_manifest:
        video_paths, para_paths = load_manifest_pairs(test_manifest)
    else:
        root = Path(config["dataset"]["test"]["test_root"])
        video_subdir = config["misc_dir"]["video_subdir"]
        norm_subdir = config["misc_dir"]["norm_subdir"]
        video_paths = sorted(str(p) for p in (root / video_subdir).glob("*.mp4"))
        para_paths = sorted(str(p) for p in (root / norm_subdir).glob("*.json"))

    dataset = dataset_class(
        video_paths,
        para_paths,
        float(config["dataset"]["test"]["frame_num"]),
        float(config["dataset"]["test"]["time"]),
        aug_bool=False,
        visc_class=int(config["model"]["transformer"]["class"]),
    )
    indexed = IndexedDataset(dataset)
    indices = list(range(len(indexed)))
    if max_samples is not None:
        indices = indices[:max_samples]
    local_indices = indices[rank::world_size]
    return DataLoader(Subset(indexed, local_indices), batch_size=batch_size, shuffle=False, num_workers=0)


def strip_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def register_final_attention_hook(model: torch.nn.Module) -> tuple[list[Any], dict[str, torch.Tensor | None]]:
    captured: dict[str, torch.Tensor | None] = {"last": None}
    handles = []

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, (tuple, list)):
            candidate = output[-1]
        else:
            candidate = output
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
    _, time_steps, _, height, width = frames.shape
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


def normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)


def entropy_norm(probs: np.ndarray) -> float:
    probs = probs.astype(np.float64)
    total = float(np.nansum(probs))
    if total <= 0:
        return 0.0
    probs = probs / total
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / math.log(probs.size))


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
    groups = temporal.shape[0]
    thirds = np.array_split(np.arange(groups), 3)
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
        center_dist = math.hypot(cy - center_y, cx - center_x) / max_dist if max_dist > 0 else 0.0
    else:
        center_dist = 0.0

    return {
        "temporal_entropy": entropy_norm(temporal),
        "spatial_entropy": entropy_norm(spatial),
        "temporal_peak_idx": float(np.nanargmax(temporal_probs)),
        "temporal_peak_frac": float(np.nanmax(temporal_probs)),
        "early_mass": early,
        "mid_mass": mid,
        "late_mass": late,
        "center_distance": center_dist,
        "top10_spatial_mass": top10_mass,
    }


def plot_panels(run_dir: Path, records: list[dict[str, Any]], volume_dir: Path) -> None:
    panel_dir = run_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "correct": [record for record in records if record["correct"]],
        "wrong": [record for record in records if not record["correct"]],
        "class1_true": [record for record in records if record["true"] == 1],
        "class2_true": [record for record in records if record["true"] == 2],
        "class3_true": [record for record in records if record["true"] == 3],
    }

    fig, axes = plt.subplots(2, len(groups), figsize=(3.2 * len(groups), 6.2))
    if len(groups) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, (group_name, group_records) in enumerate(groups.items()):
        ax_map = axes[0, col]
        ax_time = axes[1, col]
        if not group_records:
            ax_map.set_title(f"{group_name}\nno samples")
            ax_map.axis("off")
            ax_time.axis("off")
            continue
        volumes = [np.load(volume_dir / record["volume_file"]) for record in group_records]
        max_g = max(volume.shape[2] for volume in volumes)
        padded = []
        for volume in volumes:
            out = np.full((volume.shape[0], volume.shape[1], max_g), np.nan, dtype=np.float32)
            out[:, :, : volume.shape[2]] = volume
            padded.append(out)
        mean_volume = np.nanmean(np.stack(padded, axis=0), axis=0)
        spatial = normalize(np.nanmean(normalize(mean_volume), axis=2))
        temporal = normalize(np.nanmean(normalize(mean_volume), axis=(0, 1)))

        ax_map.imshow(spatial, cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
        ax_map.set_title(f"{group_name}\nn={len(group_records)}")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_time.plot(np.arange(len(temporal)), temporal, marker="o", lw=1.2)
        ax_time.set_ylim(-0.05, 1.05)
        ax_time.set_xlabel("tubelet time")
        ax_time.set_ylabel("mean attention")
        ax_time.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(panel_dir / "attention_summary_panels.png", dpi=180)
    plt.close(fig)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(records),
        "accuracy": float(np.mean([record["correct"] for record in records])) if records else None,
    }
    for prefix, group in {
        "all": records,
        "correct": [record for record in records if record["correct"]],
        "wrong": [record for record in records if not record["correct"]],
        "class1_true": [record for record in records if record["true"] == 1],
        "class2_true": [record for record in records if record["true"] == 2],
        "class3_true": [record for record in records if record["true"] == 3],
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
        ]:
            values = [record[metric] for record in group]
            summary[f"{prefix}_{metric}_mean"] = float(np.mean(values)) if values else None
    return summary


def run_one(
    run_spec: dict[str, str],
    output_root: Path,
    device: torch.device,
    max_samples: int | None,
    batch_size: int,
    rank: int,
    world_size: int,
) -> dict[str, Any] | None:
    run_name = run_spec["name"]
    with open(run_spec["config"], "r") as file:
        config = yaml.safe_load(file)
    config["train_settings"]["train_bool"] = False
    config["train_settings"]["test_bool"] = True
    config["train_settings"]["attn_bool"] = True

    checkpoint = Path(config["misc_dir"]["ckpt_root"]) / run_spec["checkpoint"]
    model = make_model(config, device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(strip_state_dict(state), strict=False)
    model.eval()
    handles, captured = register_final_attention_hook(model)
    loader = make_test_loader(config, batch_size, rank, world_size, max_samples)

    run_dir = output_root / run_name
    volume_dir = run_dir / "volumes"
    if rank == 0:
        volume_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(world_size, device)
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample_indices, frames, _parameters, hotvector, names, rpm_idx, pattern in loader:
            frames = frames.to(device)
            rpm_idx = rpm_idx.to(device).long().squeeze(-1)
            pattern = pattern.to(device)
            captured["last"] = None
            logits = model(frames, rpm_idx, pattern)
            attn = captured["last"]
            if attn is None:
                raise RuntimeError(f"No attention captured for {run_name}.")
            preds = logits.argmax(dim=1).detach().cpu().tolist()
            labels = hotvector.detach().cpu().view(-1).long().tolist()
            name_list = [str(name) for name in names]
            for batch_idx, (sample_idx, true, pred, name) in enumerate(
                zip(sample_indices.tolist(), labels, preds, name_list)
            ):
                volume = attention_volume(attn, frames, sample_idx=batch_idx)
                volume_file = f"{sample_idx:04d}_{name}_class{true}_pred{pred}_cls_attn_vol.npy"
                np.save(volume_dir / volume_file, volume.astype(np.float32))
                record = {
                    "idx": int(sample_idx),
                    "name": name,
                    "true": int(true),
                    "pred": int(pred),
                    "correct": int(pred) == int(true),
                    "volume_file": volume_file,
                }
                record.update(volume_stats(volume))
                records.append(record)

    for handle in handles:
        handle.remove()
    records = sorted(gather_records(records, world_size), key=lambda item: item["idx"])
    distributed_barrier(world_size, device)
    if rank != 0:
        return None

    with open(run_dir / "sample_attention_metrics.csv", "w", newline="") as file:
        fieldnames = list(records[0].keys()) if records else ["idx"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = summarize_records(records)
    summary.update(
        {
            "run_name": run_name,
            "checkpoint": str(checkpoint),
            "config": run_spec["config"],
            "max_samples": max_samples,
        }
    )
    with open(run_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)
    plot_panels(run_dir, records, volume_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/rebuild_reproduction/attention_993")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    rank, world_size, _local_rank, device = init_distributed(args.device)

    run_specs = [
        {
            "name": "realonly_993_300epoch",
            "config": "configs/rebuild/realonly_993.yaml",
            "checkpoint": "repro_realonly_993.pth",
        },
        {
            "name": "realonly_993_lrhold",
            "config": "configs/rebuild/retries/realonly_993_microbatch_lrhold.yaml",
            "checkpoint": "repro_realonly_993_microbatch_lrhold.pth",
        },
        {
            "name": "transfer_993_lrhold",
            "config": "configs/rebuild/retries/transfer_993_microbatch_lrhold.yaml",
            "checkpoint": "repro_transfer_993_microbatch_lrhold.pth",
        },
        {
            "name": "realonly_993_30ep_microbatch",
            "config": "configs/rebuild/realonly_993.yaml",
            "checkpoint": "repro_realonly_993_microbatch.pth",
        },
        {
            "name": "transfer_993_30ep_microbatch",
            "config": "configs/rebuild/transfer_993.yaml",
            "checkpoint": "repro_transfer_993_microbatch.pth",
        },
    ]

    output_root = Path(args.output_root)
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
    distributed_barrier(world_size, device)
    summaries = [
        summary
        for summary in (
            run_one(spec, output_root, device, args.max_samples, args.batch_size, rank, world_size)
            for spec in run_specs
        )
        if summary is not None
    ]
    if rank != 0:
        cleanup_distributed(world_size)
        return
    with open(output_root / "summary.json", "w") as file:
        json.dump(summaries, file, indent=2)
    with open(output_root / "summary.csv", "w", newline="") as file:
        fieldnames = sorted({key for summary in summaries for key in summary.keys()})
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {output_root / 'summary.json'}")
    print(f"Wrote {output_root / 'summary.csv'}")
    cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
