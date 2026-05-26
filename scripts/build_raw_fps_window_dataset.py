#!/usr/bin/env python3
"""Build FPS-aware raw-real temporal-window manifests and optional clips."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_MANIFEST = ROOT / "dataset/RealArchive/real_20rpm_increment_2500/raw_config_manifest.json"
FINAL_ROOT = ROOT / "dataset/RealArchive/real_20rpm_increment_2500"
EXISTING_TRAIN_ROOT = ROOT / "dataset/RealArchive/train_993_wo_pat2"
EXISTING_TEST_ROOT = ROOT / "dataset/RealArchive/test_1000_wo_pat2"
BACKGROUND_SOURCE = EXISTING_TRAIN_ROOT / "backgrounds"
OUTPUT_ROOT = ROOT / "outputs/rebuild_reproduction/derived_datasets/raw_fps_benchmark"
CONFIG_ROOT = ROOT / "configs/rebuild/raw_fps"

LEGACY_SPANS = [30, 36, 42, 50]
NATIVE_DURATIONS = [1.25, 1.50, 1.75, 2.08]
MODEL_FRAMES = 30
SEED = 37
WANDB_ENTITY = "jongwonsohn-seoul-national-university"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "re-rebuild-viscnet")


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def parse_phase_offsets(value: str, sample_stride: int) -> list[int]:
    if sample_stride < 1:
        return [0]
    if value == "all":
        return list(range(sample_stride))
    offsets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not offsets:
        raise ValueError("--phase-offsets must provide at least one offset")
    invalid = [offset for offset in offsets if offset < 0 or offset >= sample_stride]
    if invalid:
        raise ValueError(f"phase offsets must be in [0, {sample_stride - 1}], got {invalid}")
    return offsets


def load_raw_records(path: Path) -> list[dict]:
    records = read_json(path)
    if isinstance(records, dict):
        records = records["samples"]
    by_stem = {}
    for record in records:
        if not record.get("use_in_training", record.get("usable", True)):
            continue
        stem = record["target_video_stem"]
        norm_path = FINAL_ROOT / "parametersNorm" / f"{stem}.json"
        param_path = FINAL_ROOT / "parameters" / f"{stem}.json"
        if not norm_path.exists() or not param_path.exists():
            continue
        row = dict(record)
        row["source_id"] = f"{row['source_group']}:{Path(row['source_file']).stem}"
        row["parameters_norm_path"] = relative(norm_path)
        row["parameters_path"] = relative(param_path)
        by_stem[stem] = row
    return [by_stem[key] for key in sorted(by_stem, key=lambda stem: by_stem[stem]["combined_index"])]


def stems_from_root(root: Path) -> set[str]:
    return {path.stem for path in (root / "parametersNorm").glob("*.json")}


def existing_source_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    by_stem = {record["target_video_stem"]: record for record in records}
    train = [by_stem[stem] for stem in sorted(stems_from_root(EXISTING_TRAIN_ROOT)) if stem in by_stem]
    test = [by_stem[stem] for stem in sorted(stems_from_root(EXISTING_TEST_ROOT)) if stem in by_stem]
    return train, test


def take_round_robin(grouped: dict[tuple[int, int], list[dict]], count: int) -> list[dict]:
    keys = sorted(grouped)
    selected = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if grouped[key] and len(selected) < count:
                selected.append(grouped[key].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"Could only select {len(selected)} records, requested {count}")
    return sorted(selected, key=lambda record: record["combined_index"])


def ratio_source_split(records: list[dict], train_count: int = 1500, test_count: int = 500) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    rng = random.Random(SEED)
    for record in records:
        grouped[(int(record["viscosity_block_index"]), int(record["RPM_index"]))].append(record)
    for rows in grouped.values():
        rows.sort(key=lambda record: record["combined_index"])
        rng.shuffle(rows)
    test = take_round_robin(grouped, test_count)
    train = take_round_robin(grouped, train_count)
    return train, test


def fps_bin(fps: float) -> str:
    if 23.0 <= fps < 25.0:
        return "~24fps"
    if 29.0 <= fps < 31.0:
        return "~30fps"
    return "outlier_or_unknown"


def window_specs(
    record: dict,
    policy: str,
    windows_per_span: int,
    *,
    sample_stride: int,
    phase_offsets: list[int],
) -> list[dict]:
    source_fps = float(record["source_fps"])
    frame_count = int(record["source_frame_count"])
    specs = []
    if policy == "legacy10fps":
        candidates = [(float(span) / 10.0, int(span), 10.0, span) for span in LEGACY_SPANS]
    elif policy == "nativefps":
        candidates = []
        for duration in NATIVE_DURATIONS:
            source_span = max(MODEL_FRAMES, int(round(duration * source_fps)))
            derived_fps = MODEL_FRAMES / duration
            candidates.append((duration, source_span, derived_fps, duration))
    else:
        raise ValueError(f"Unsupported FPS policy: {policy}")

    for duration, source_span, derived_fps, candidate in candidates:
        if source_span > frame_count:
            continue
        max_start = frame_count - source_span
        if windows_per_span <= 1 or max_start == 0:
            starts = [max_start // 2]
        else:
            starts = sorted({int(round(max_start * idx / (windows_per_span - 1))) for idx in range(windows_per_span)})
        for start in starts:
            for phase_offset in phase_offsets:
                effective_start = start + phase_offset
                end = start + source_span
                spec = {
                    "fps_policy": policy,
                    "window_candidate": candidate,
                    "window_start_frame": effective_start,
                    "window_end_frame": end,
                    "source_frame_span": source_span,
                    "duration_seconds": duration,
                    "window_duration_seconds": duration,
                    "derived_fps": derived_fps,
                    "sample_stride": sample_stride,
                    "phase_offset": phase_offset,
                }
                if sample_stride > 0:
                    indices = [effective_start + idx * sample_stride for idx in range(MODEL_FRAMES)]
                    if indices[-1] >= end:
                        continue
                    spec["source_frame_indices"] = indices
                    spec["derived_fps"] = source_fps / sample_stride
                    spec["duration_seconds"] = (indices[-1] - indices[0] + 1) / source_fps
                    spec["window_duration_seconds"] = spec["duration_seconds"]
                specs.append(spec)
    return specs


def frame_indices(start: int, end: int, count: int) -> list[int]:
    if count <= 1:
        return [start]
    return [min(end - 1, int(round(start + (end - 1 - start) * idx / (count - 1)))) for idx in range(count)]


def load_frames(path: Path) -> list:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open raw video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def crop_resize(frame, image_size: int):
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    cropped = frame[top : top + side, left : left + side]
    interpolation = cv2.INTER_AREA if side >= image_size else cv2.INTER_LINEAR
    return cv2.resize(cropped, (image_size, image_size), interpolation=interpolation)


def write_clip(path: Path, frames: list, fps: float, image_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (image_size, image_size))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for frame in frames:
        writer.write(crop_resize(frame, image_size))
    writer.release()


def ensure_backgrounds(output_root: Path) -> None:
    target = output_root / "backgrounds"
    if target.exists() or target.is_symlink():
        return
    try:
        os.symlink(os.path.relpath(BACKGROUND_SOURCE, output_root), target, target_is_directory=True)
    except OSError:
        shutil.copytree(BACKGROUND_SOURCE, target)


def build_split_manifest(
    records: list[dict],
    split_name: str,
    split_part: str,
    policy: str,
    image_size: int,
    output_root: Path,
    *,
    materialize: bool,
    windows_per_span: int,
    sample_stride: int,
    phase_offsets: list[int],
) -> dict:
    split_root = output_root / split_name / policy / f"image{image_size}" / split_part
    video_root = split_root / "videos"
    ensure_backgrounds(split_root)
    samples = []
    total_windows = 0
    skipped_sources = []
    for source_index, record in enumerate(records):
        specs = window_specs(
            record,
            policy,
            windows_per_span,
            sample_stride=sample_stride,
            phase_offsets=phase_offsets,
        )
        if not specs:
            skipped_sources.append(record["source_id"])
            continue
        source_path = ROOT / record["source_path"]
        frames = load_frames(source_path) if materialize else None
        for window_index, spec in enumerate(specs):
            clip_name = f"{record['target_video_stem']}_{policy}_w{window_index:02d}.mp4"
            clip_path = video_root / clip_name
            if materialize:
                indices = spec.get("source_frame_indices") or frame_indices(
                    spec["window_start_frame"], spec["window_end_frame"], MODEL_FRAMES
                )
                clip_frames = [frames[index] for index in indices]
                write_clip(clip_path, clip_frames, float(spec["derived_fps"]), image_size)
            sample = {
                "video_path": relative(clip_path),
                "parameters_norm_path": record["parameters_norm_path"],
                "parameters_path": record["parameters_path"],
                "source_id": record["source_id"],
                "source_group": record["source_group"],
                "source_path": record["source_path"],
                "source_index": source_index,
                "source_fps": record["source_fps"],
                "source_fps_bin": fps_bin(float(record["source_fps"])),
                "source_frame_count": record["source_frame_count"],
                "source_duration_seconds": record["source_duration_seconds"],
                "source_width": record["source_width"],
                "source_height": record["source_height"],
                "target_video_stem": record["target_video_stem"],
                "visc_index": record["viscosity_block_index"],
                "rpm_idx": record["RPM_index"],
                **spec,
            }
            samples.append(sample)
            total_windows += 1
    manifest_path = split_root / "manifest.json"
    summary_path = split_root / "summary.json"
    payload = {
        "seed": SEED,
        "split_name": split_name,
        "split_part": split_part,
        "fps_policy": policy,
        "image_size": image_size,
        "model_num_frames": MODEL_FRAMES,
        "windows_per_span": windows_per_span,
        "sample_stride": sample_stride,
        "phase_offsets": phase_offsets,
        "source_count": len(records),
        "sample_count": len(samples),
        "materialized_clips": materialize,
        "samples": samples,
    }
    write_json(manifest_path, payload)
    summary = {
        "manifest": relative(manifest_path),
        "source_count": len(records),
        "sample_count": len(samples),
        "skipped_source_count": len(skipped_sources),
        "skipped_sources": skipped_sources[:50],
        "fps_bins": dict(sorted({sample["source_fps_bin"]: 0 for sample in samples}.items())),
    }
    for sample in samples:
        summary["fps_bins"][sample["source_fps_bin"]] = summary["fps_bins"].get(sample["source_fps_bin"], 0) + 1
    write_json(summary_path, summary)
    return summary | {"manifest": relative(manifest_path), "total_windows": total_windows}


def base_config(name: str, train_manifest: str, test_manifest: str, image_size: int) -> dict:
    dataset_class = "VideoDatasetReal336" if image_size == 336 else "VideoDatasetReal"
    batch_size = 4 if image_size == 336 else 32
    eval_batch_size = 8 if image_size == 336 else 64
    return {
        "project": WANDB_PROJECT,
        "entity": os.environ.get("WANDB_ENTITY", WANDB_ENTITY),
        "name": name,
        "version": "v0",
        "train_settings": {
            "num_workers": 0,
            "seed": 1205,
            "classification": True,
            "gmm_bool": False,
            "watch_bool": False,
            "train_bool": True,
            "test_bool": True,
            "attn_bool": False,
            "val_test_bool": True,
            "sanity_check_bool": False,
        },
        "dataset": {
            "train": {
                "train_root": str(Path(train_manifest).parent),
                "use_all_samples": True,
                "frame_num": 6,
                "time": 5,
                "rpm_class": 10,
                "dataloader": {
                    "dataloader": dataset_class,
                    "batch_size": batch_size,
                    "aug_bool": False,
                    "test_size": 1e-6,
                    "random_state": SEED,
                },
                "manifest": train_manifest,
            },
            "test": {
                "test_root": str(Path(test_manifest).parent),
                "frame_num": 6,
                "time": 5,
                "rpm_class": 10,
                "dataloader": {
                    "dataloader": dataset_class,
                    "batch_size": eval_batch_size,
                    "aug_bool": False,
                    "test_size": 1e-6,
                    "random_state": SEED,
                },
                "manifest": test_manifest,
            },
            "preprocess": {"scaler": "interscaler", "descaler": "interdescaler"},
        },
        "model": {
            "transformer_bool": True,
            "transformer": {
                "encoder": "VivitEmbed",
                "class": 10,
                "num_frames": MODEL_FRAMES,
                "image_size": image_size,
            },
            "embeddings": {"rpm_bool": True, "pat_bool": False},
            "cnn": {
                "encoder": "Resnet34LSTMEMBED",
                "cnn_train": True,
                "cnn": "resnet34",
                "lstm_size": 256,
                "lstm_layers": 5,
                "output_size": 3,
                "drop_rate": 0.0,
                "embedding_size": 512,
                "embed_weight": 0.0,
            },
            "gmm": {"gmm_num": 3},
        },
        "training": {
            "curr_bool": False,
            "curr_ckpt": "repro_synthetic_pretrain_sph35000.pth",
            "checkpoint_name": f"{name}.pth",
            "num_epochs": 50,
            "loss": "CE",
            "label_smoothing": 0.0,
            "optimizer": {
                "optim_class": "AdamW",
                "scheduler_class": "CosineAnnealingLR",
                "schedule_policy": "warmup_hold_cosine",
                "warmup_epochs": 1,
                "warmup_start_factor": 0.5,
                "lr_hold_epochs": 10,
                "lr": 1e-5,
                "eta_min": 1e-10,
                "weight_decay": 1e-2,
                "patience": 12,
            },
            "acceptance": {
                "allow_frame_count_override": True,
                "allow_derived_window_dataset": True,
                "note": "raw real-only FPS-aware variable-window benchmark; not a paper reproduction pass/fail run",
            },
        },
        "misc_dir": {
            "ckpt_root": "outputs/rebuild_reproduction/checkpoints",
            "output_root": f"outputs/rebuild_reproduction/{name}",
            "video_subdir": "videos",
            "para_subdir": "parameters",
            "norm_subdir": "parametersNorm",
        },
        "evaluation": {"report_source_averaged": True, "report_fps_stratified": True},
    }


def write_configs(manifest_index: dict[tuple[str, str, str, int], str]) -> list[str]:
    paths = []
    for (split_name, split_part, policy, image_size), manifest in sorted(manifest_index.items()):
        if split_part != "train":
            continue
        test_manifest = manifest_index[(split_name, "test", policy, image_size)]
        name = f"rawfps_realonly_{split_name}_{policy}_{image_size}px"
        path = CONFIG_ROOT / f"{name}.yaml"
        write_yaml(path, base_config(name, manifest, test_manifest, image_size))
        paths.append(relative(path))
    return paths


def validate_splits(splits: dict[str, tuple[list[dict], list[dict]]]) -> None:
    expected = {"existing993_1000": (993, 1000), "ratio1500_500": (1500, 500)}
    for name, (train, test) in splits.items():
        train_expected, test_expected = expected[name]
        if len(train) != train_expected or len(test) != test_expected:
            raise RuntimeError(f"{name} expected {train_expected}/{test_expected}, got {len(train)}/{len(test)}")
        train_ids = {record["source_id"] for record in train}
        test_ids = {record["source_id"] for record in test}
        overlap = train_ids & test_ids
        if overlap:
            raise RuntimeError(f"{name} has train/test source overlap: {sorted(overlap)[:10]}")


def build(args: argparse.Namespace) -> dict:
    records = load_raw_records(ROOT / args.raw_manifest)
    if len(records) < 2000:
        raise RuntimeError(f"Need at least 2000 usable raw records, found {len(records)}")
    splits = {
        "existing993_1000": existing_source_split(records),
        "ratio1500_500": ratio_source_split(records),
    }
    validate_splits(splits)
    phase_offsets = parse_phase_offsets(args.phase_offsets, args.sample_stride)
    if args.max_sources_per_part is not None:
        limit = max(1, int(args.max_sources_per_part))
        splits = {name: (train[:limit], test[:limit]) for name, (train, test) in splits.items()}

    output_root = ROOT / args.output_root
    manifest_index = {}
    summaries = {}
    for split_name, parts in splits.items():
        for split_part, split_records in zip(["train", "test"], parts, strict=True):
            for policy in args.fps_policy:
                summary = build_split_manifest(
                    split_records,
                    split_name,
                    split_part,
                    policy,
                    args.image_size,
                    output_root,
                    materialize=args.materialize_clips,
                    windows_per_span=args.windows_per_span,
                    sample_stride=args.sample_stride,
                    phase_offsets=phase_offsets,
                )
                summaries[f"{split_name}/{policy}/{args.image_size}px/{split_part}"] = summary
                manifest_index[(split_name, split_part, policy, args.image_size)] = summary["manifest"]

    written_configs = write_configs(manifest_index) if args.write_configs else []
    report = {
        "raw_manifest": args.raw_manifest,
        "output_root": args.output_root,
        "materialized_clips": args.materialize_clips,
        "image_size": args.image_size,
        "model_num_frames": MODEL_FRAMES,
        "written_configs": written_configs,
        "summaries": summaries,
    }
    write_json(output_root / f"build_report_image{args.image_size}.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", default=relative(RAW_MANIFEST))
    parser.add_argument("--output-root", default=relative(OUTPUT_ROOT))
    parser.add_argument("--image-size", type=int, choices=[224, 336], default=224)
    parser.add_argument("--fps-policy", nargs="+", choices=["legacy10fps", "nativefps"], default=["legacy10fps", "nativefps"])
    parser.add_argument("--windows-per-span", type=int, default=1)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=0,
        help="Optional fixed source-frame stride. Default 0 preserves linspace sampling to 30 frames.",
    )
    parser.add_argument(
        "--phase-offsets",
        default="0",
        help='Comma-separated offsets, or "all" for offsets 0..sample_stride-1.',
    )
    parser.add_argument("--max-sources-per-part", type=int)
    parser.add_argument("--materialize-clips", action="store_true")
    parser.add_argument("--write-configs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
