#!/usr/bin/env python3
"""Generate paper-reproduction configs and deterministic real-data manifests."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "rebuild"
MANIFEST_ROOT = CONFIG_ROOT / "manifests"
REFERENCE_PATH = CONFIG_ROOT / "reference_metrics.json"

REAL_TRAIN_ROOT = Path("dataset/RealArchive/train_993_wo_pat2")
REAL_TEST_ROOT = Path("dataset/RealArchive/test_1000_wo_pat2")
SYNTH_ROOT = Path("dataset/CFDArchive/sph_35000")
CKPT_ROOT = "outputs/rebuild_reproduction/checkpoints"
OUTPUT_ROOT = "outputs/rebuild_reproduction"
SYNTH_CKPT = "repro_synthetic_pretrain_sph35000.pth"
SEED = 37
SIZES = [300, 400, 500, 600, 700, 800, 900, 993]
BATCH_SIZE_PER_GPU = 10
OPTIMIZER_MICROBATCH_SIZE = 1
NUM_WORKERS = 0
LONG_HORIZON_NUM_EPOCHS = 300
MICROBATCH_NUM_EPOCHS = 30
MICROBATCH_SUFFIX = "_microbatch"
WANDB_ENTITY = "jongwonsohn-seoul-national-university"


REFERENCE_METRICS = {
    "synthetic_pretrain": {
        "project": "sph_test_run",
        "run_id": "mffvc6vq",
        "run_name": "35000_weightmaking_smallerVivit_layer10_1024_v0",
        "accuracy": None,
        "train_loss": 0.3351,
        "val_loss": 0.6146,
        "note": "Reference W&B used dataset/CFDArchive/sph_realvisc_diffback_35000, which is absent in this repo.",
    },
    "transfer_curve": {
        "300": {"project": "dataefficiency", "run_id": "2fn4fl9b", "accuracy": 0.6020},
        "400": {"project": "dataefficiency", "run_id": "ph6rwzbz", "accuracy": 0.6930},
        "500": {"project": "dataefficiency", "run_id": "jvqqegb9", "accuracy": 0.7160},
        "600": {"project": "dataefficiency", "run_id": "1gqccpob", "accuracy": 0.7350},
        "700": {"project": "dataefficiency", "run_id": "3mn3dnfp", "accuracy": 0.7720},
        "800": {"project": "dataefficiency", "run_id": "rznnlkan", "accuracy": 0.8010},
        "900": {"project": "dataefficiency", "run_id": "zhc7d47c", "accuracy": 0.7970},
        "993": {"project": "dataefficiency", "run_id": "6r1p7fet", "accuracy": 0.8090},
    },
    "real_only_curve": {
        "300": {"project": "dataefficiency", "run_id": "g5nheksd", "accuracy": 0.3180},
        "400": {"project": "dataefficiency", "run_id": "883f5m6k", "accuracy": 0.4210},
        "500": {"project": "dataefficiency", "run_id": "dp5p39bc", "accuracy": 0.4860},
        "600": {"project": "dataefficiency", "run_id": "utgbwbeu", "accuracy": 0.5210},
        "700": {"project": "dataefficiency", "run_id": "l4ibkqqg", "accuracy": 0.6010},
        "800": {"project": "dataefficiency", "run_id": "hq4o5lfo", "accuracy": 0.6460},
        "900": {"project": "dataefficiency", "run_id": "l7zrpyt7", "accuracy": 0.6950},
        "993": {"project": "dataefficiency", "run_id": "i58ha0w0", "accuracy": 0.7230},
    },
    "pattern_reference": {
        "project": "sph_test_run",
        "run_id": "p5iarxc7",
        "run_name": "5thpatVal_finetuning_0929_v0",
        "accuracy": 0.8421,
        "note": "Old run used unavailable 1back/4back split names and unavailable diffback checkpoint.",
    },
}


def read_records(source_root: Path) -> list[dict]:
    records = []
    video_root = ROOT / source_root / "videos"
    norm_root = ROOT / source_root / "parametersNorm"
    for para_path in sorted(norm_root.glob("*.json")):
        video_path = video_root / f"{para_path.stem}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        data = json.loads(para_path.read_text())
        render_label = para_path.stem.split("_render")[-1] if "_render" in para_path.stem else ""
        records.append(
            {
                "video_path": str(video_path.relative_to(ROOT)),
                "parameters_norm_path": str(para_path.relative_to(ROOT)),
                "visc_index": int(data["visc_index"]),
                "rpm_idx": int(data["rpm_idx"]),
                "render_label": render_label,
            }
        )
    return records


def stratified_subset(records: list[dict], size: int) -> list[dict]:
    if size == len(records):
        return sorted(records, key=lambda row: row["parameters_norm_path"])
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["visc_index"], record["rpm_idx"])].append(record)
    rng = random.Random(SEED)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["parameters_norm_path"])
        rng.shuffle(rows)
    selected = []
    keys = sorted(grouped)
    while len(selected) < size:
        progressed = False
        for key in keys:
            if grouped[key] and len(selected) < size:
                selected.append(grouped[key].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"Could not build subset of {size}; only selected {len(selected)}")
    return sorted(selected, key=lambda row: row["parameters_norm_path"])


def write_manifest(size: int, records: list[dict]) -> str:
    manifest_path = MANIFEST_ROOT / f"real_train_{size}.json"
    payload = {
        "source_root": str(REAL_TRAIN_ROOT),
        "size": size,
        "seed": SEED,
        "method": "round-robin stratified by visc_index then rpm_idx from train_993_wo_pat2",
        "samples": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return str(manifest_path.relative_to(ROOT))


def base_config(
    name: str,
    checkpoint_name: str,
    *,
    num_epochs: int,
    optimizer_microbatch_size: int | None = None,
) -> dict:
    training = {
        "curr_bool": False,
        "curr_ckpt": SYNTH_CKPT,
        "checkpoint_name": checkpoint_name,
        "num_epochs": num_epochs,
        "loss": "CE",
        "label_smoothing": 0.0,
        "optimizer": {
            "optim_class": "AdamW",
            "scheduler_class": "CosineAnnealingLR",
            "lr": 1e-5,
            "eta_min": 1e-10,
            "weight_decay": 1e-2,
            "patience": 10,
        },
    }
    if optimizer_microbatch_size is not None:
        training["update_density"] = {"optimizer_microbatch_size": optimizer_microbatch_size}

    return {
        "project": "viscnet-rebuild",
        "entity": WANDB_ENTITY,
        "name": name,
        "version": "v0",
        "train_settings": {
            "num_workers": NUM_WORKERS,
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
                "train_root": str(REAL_TRAIN_ROOT),
                "frame_num": 10,
                "time": 5,
                "rpm_class": 10,
                "dataloader": {
                    "dataloader": "VideoDatasetReal",
                    "batch_size": BATCH_SIZE_PER_GPU,
                    "aug_bool": False,
                    "test_size": 1e-6,
                    "random_state": 37,
                },
            },
            "test": {
                "test_root": str(REAL_TEST_ROOT),
                "frame_num": 10,
                "time": 5,
                "rpm_class": 10,
                "dataloader": {
                    "dataloader": "VideoDatasetReal",
                    "batch_size": BATCH_SIZE_PER_GPU,
                    "aug_bool": False,
                    "test_size": 1e-6,
                    "random_state": 37,
                },
            },
            "preprocess": {"scaler": "interscaler", "descaler": "interdescaler"},
        },
        "model": {
            "transformer_bool": True,
            "transformer": {"encoder": "VivitEmbed", "class": 10},
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
        "training": training,
        "misc_dir": {
            "ckpt_root": CKPT_ROOT,
            "output_root": f"{OUTPUT_ROOT}/{name}",
            "video_subdir": "videos",
            "para_subdir": "parameters",
            "norm_subdir": "parametersNorm",
        },
    }


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def make_configs(manifests: dict[int, str]) -> None:
    synth = base_config(
        "repro_synthetic_pretrain_sph35000",
        SYNTH_CKPT,
        num_epochs=LONG_HORIZON_NUM_EPOCHS,
    )
    synth["train_settings"]["test_bool"] = False
    synth["train_settings"]["val_test_bool"] = False
    synth["train_settings"]["sanity_check_bool"] = False
    synth["dataset"]["train"]["train_root"] = str(SYNTH_ROOT)
    synth["dataset"]["train"]["dataloader"]["dataloader"] = "VideoDataset"
    synth["dataset"]["train"]["dataloader"]["test_size"] = 0.2
    write_yaml(CONFIG_ROOT / "synthetic_pretrain.yaml", synth)

    for size in SIZES:
        real_only = base_config(
            f"repro_realonly_{size}{MICROBATCH_SUFFIX}",
            f"repro_realonly_{size}{MICROBATCH_SUFFIX}.pth",
            num_epochs=MICROBATCH_NUM_EPOCHS,
            optimizer_microbatch_size=OPTIMIZER_MICROBATCH_SIZE,
        )
        real_only["dataset"]["train"]["manifest"] = manifests[size]
        write_yaml(CONFIG_ROOT / f"realonly_{size}.yaml", real_only)

        transfer = base_config(
            f"repro_transfer_{size}{MICROBATCH_SUFFIX}",
            f"repro_transfer_{size}{MICROBATCH_SUFFIX}.pth",
            num_epochs=MICROBATCH_NUM_EPOCHS,
            optimizer_microbatch_size=OPTIMIZER_MICROBATCH_SIZE,
        )
        transfer["dataset"]["train"]["manifest"] = manifests[size]
        transfer["training"]["curr_bool"] = True
        transfer["training"]["curr_ckpt"] = SYNTH_CKPT
        write_yaml(CONFIG_ROOT / f"transfer_{size}.yaml", transfer)

    pattern_pairs = {
        "train1234_test5": (
            "dataset/RealArchive/real_20rpm_increment_train1234",
            "dataset/RealArchive/real_20rpm_increment_test5",
        ),
        "train134_test5": (
            "dataset/RealArchive/real_20rpm_increment_train134",
            "dataset/RealArchive/real_20rpm_increment_test5",
        ),
        "train135_test4": (
            "dataset/RealArchive/real_20rpm_increment_train135",
            "dataset/RealArchive/real_20rpm_increment_test4",
        ),
        "train345_test1": (
            "dataset/RealArchive/real_20rpm_increment_train345",
            "dataset/RealArchive/real_20rpm_increment_test1",
        ),
    }
    for suffix, (train_root, test_root) in pattern_pairs.items():
        cfg = base_config(
            f"repro_pattern_{suffix}{MICROBATCH_SUFFIX}",
            f"repro_pattern_{suffix}{MICROBATCH_SUFFIX}.pth",
            num_epochs=MICROBATCH_NUM_EPOCHS,
            optimizer_microbatch_size=OPTIMIZER_MICROBATCH_SIZE,
        )
        cfg["dataset"]["train"]["train_root"] = train_root
        cfg["dataset"]["test"]["test_root"] = test_root
        cfg["dataset"]["train"]["dataloader"]["aug_bool"] = True
        cfg["training"]["curr_bool"] = True
        cfg["training"]["curr_ckpt"] = SYNTH_CKPT
        write_yaml(CONFIG_ROOT / f"pattern_{suffix}.yaml", cfg)


def main() -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / OUTPUT_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    records = read_records(REAL_TRAIN_ROOT)
    manifests = {}
    for size in SIZES:
        manifests[size] = write_manifest(size, stratified_subset(records, size))
    make_configs(manifests)
    REFERENCE_PATH.write_text(json.dumps(REFERENCE_METRICS, indent=2) + "\n")
    print(f"Wrote rebuild configs to {CONFIG_ROOT.relative_to(ROOT)}")
    print(f"Wrote manifests to {MANIFEST_ROOT.relative_to(ROOT)}")
    print(f"Wrote references to {REFERENCE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
