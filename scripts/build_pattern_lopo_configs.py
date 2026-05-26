#!/usr/bin/env python3
"""Build leave-one-pattern-out real-video splits and training configs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = [
    ROOT / "dataset/RealArchive/train_993_wo_pat2",
    ROOT / "dataset/RealArchive/test_1000_wo_pat2",
]
OUTPUT_ROOT = ROOT / "outputs/rebuild_reproduction/derived_datasets/pattern_lopo_1993"
CONFIG_ROOT = ROOT / "configs/rebuild/pattern_lopo"
SYNTHETIC30_CKPT = "repro_synthetic_pretrain_window30_batch8_ep50.pth"


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def infer_pattern_id(render: str) -> int:
    if render in set("ABCDEFGHIJ"):
        return 1
    if render in set("KLMNO"):
        return 2
    if render in set("PQRST"):
        return 3
    if render in set("UVWXY"):
        return 4
    raise ValueError(f"Cannot infer pattern id from render tag: {render}")


def load_records() -> list[dict]:
    records = []
    seen = set()
    for root in SOURCE_ROOTS:
        for norm_path in sorted((root / "parametersNorm").glob("*.json")):
            stem = norm_path.stem
            if stem in seen:
                raise RuntimeError(f"Duplicate source stem across roots: {stem}")
            seen.add(stem)
            video_path = root / "videos" / f"{stem}.mp4"
            param_path = root / "parameters" / f"{stem}.json"
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            data = read_json(norm_path)
            render = str(data.get("RENDER") or stem.split("_render")[-1])
            records.append(
                {
                    "stem": stem,
                    "source_root": relative(root),
                    "video_path": relative(video_path),
                    "parameters_norm_path": relative(norm_path),
                    "parameters_path": relative(param_path) if param_path.exists() else "",
                    "render": render,
                    "pattern_id": infer_pattern_id(render),
                    "visc_index": int(data.get("visc_index", data.get("INDEX"))),
                    "rpm_idx": int(data.get("rpm_idx", data.get("RPM_index", 0))),
                }
            )
    return records


def symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        current = (dst.parent / os.readlink(dst)).resolve()
        if current == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        return
    os.symlink(os.path.relpath(src, dst.parent), dst)


def materialize_split(records: list[dict], split_root: Path) -> None:
    for subdir in ["videos", "parametersNorm", "parameters"]:
        (split_root / subdir).mkdir(parents=True, exist_ok=True)
    backgrounds_src = SOURCE_ROOTS[0] / "backgrounds"
    symlink_force(backgrounds_src, split_root / "backgrounds")
    samples = []
    for row in records:
        video_src = ROOT / row["video_path"]
        norm_src = ROOT / row["parameters_norm_path"]
        symlink_force(video_src, split_root / "videos" / video_src.name)
        symlink_force(norm_src, split_root / "parametersNorm" / norm_src.name)
        if row["parameters_path"]:
            param_src = ROOT / row["parameters_path"]
            symlink_force(param_src, split_root / "parameters" / param_src.name)
        sample = dict(row)
        sample["video_path"] = relative(split_root / "videos" / video_src.name)
        sample["parameters_norm_path"] = relative(split_root / "parametersNorm" / norm_src.name)
        if row["parameters_path"]:
            sample["parameters_path"] = relative(split_root / "parameters" / Path(row["parameters_path"]).name)
        samples.append(sample)
    write_json(split_root / "manifest.json", {"samples": samples})


def base_config(name: str, train_root: Path, test_root: Path, transfer: bool) -> dict:
    return {
        "project": os.environ.get("WANDB_PROJECT", "re-rebuild-viscnet"),
        "entity": os.environ.get("WANDB_ENTITY", "jongwonsohn-seoul-national-university"),
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
                "train_root": relative(train_root),
                "use_all_samples": True,
                "frame_num": 6,
                "time": 5,
                "rpm_class": 10,
                "dataloader": {
                    "dataloader": "VideoDatasetReal",
                    "batch_size": 8,
                    "aug_bool": False,
                    "test_size": 1e-6,
                    "random_state": 37,
                },
                "manifest": relative(train_root / "manifest.json"),
            },
            "test": {
                "test_root": relative(test_root),
                "frame_num": 6,
                "time": 5,
                "rpm_class": 10,
                "dataloader": {
                    "dataloader": "VideoDatasetReal",
                    "batch_size": 8,
                    "aug_bool": False,
                    "test_size": 1e-6,
                    "random_state": 37,
                },
                "manifest": relative(test_root / "manifest.json"),
            },
            "preprocess": {"scaler": "interscaler", "descaler": "interdescaler"},
        },
        "model": {
            "transformer_bool": True,
            "transformer": {"encoder": "VivitEmbed", "class": 10, "num_frames": 30, "image_size": 224},
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
            "curr_bool": transfer,
            "curr_ckpt": SYNTHETIC30_CKPT,
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
                "allow_batch_size_override": True,
                "allow_pattern_lopo_dataset": True,
                "note": "leave-one-pattern-out real-video training; diagnostic generalization run",
            },
        },
        "misc_dir": {
            "ckpt_root": "outputs/rebuild_reproduction/checkpoints",
            "output_root": f"outputs/rebuild_reproduction/{name}",
            "video_subdir": "videos",
            "para_subdir": "parameters",
            "norm_subdir": "parametersNorm",
        },
    }


def build(args: argparse.Namespace) -> dict:
    records = load_records()
    if len(records) != 1993:
        raise RuntimeError(f"Expected 1993 combined train/test records, got {len(records)}")
    by_pattern = {pattern_id: [row for row in records if row["pattern_id"] == pattern_id] for pattern_id in range(1, 5)}
    report = {
        "source_roots": [relative(path) for path in SOURCE_ROOTS],
        "total_records": len(records),
        "pattern_counts": {str(pattern_id): len(rows) for pattern_id, rows in by_pattern.items()},
        "splits": [],
    }
    config_paths = []
    for holdout in range(1, 5):
        train_records = [row for row in records if row["pattern_id"] != holdout]
        test_records = by_pattern[holdout]
        split_root = ROOT / args.output_root / f"holdout_pattern{holdout}"
        train_root = split_root / "train"
        test_root = split_root / "test"
        materialize_split(train_records, train_root)
        materialize_split(test_records, test_root)
        split_info = {
            "holdout_pattern": holdout,
            "train_count": len(train_records),
            "test_count": len(test_records),
            "train_root": relative(train_root),
            "test_root": relative(test_root),
        }
        report["splits"].append(split_info)
        for family, transfer in [("realonly", False), ("transfer_synth30", True)]:
            name = f"pattern_lopo_{family}_train_not{holdout}_test{holdout}"
            path = ROOT / args.config_root / f"{name}.yaml"
            write_yaml(path, base_config(name, train_root, test_root, transfer))
            config_paths.append(relative(path))
    report["configs"] = config_paths
    write_json(ROOT / args.output_root / "summary.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=relative(OUTPUT_ROOT))
    parser.add_argument("--config-root", default=relative(CONFIG_ROOT))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
