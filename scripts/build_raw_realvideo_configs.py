#!/usr/bin/env python3
"""Build metadata configs for the raw real-video MOV archives without touching videos."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]

DYNAMIC_VISCOSITY_CP = [
    160.4767961,
    120.2684812,
    90.13457207,
    67.5508754,
    50.62564351,
    37.94111891,
    28.43476962,
    21.31028677,
    15.97088115,
    0.89273932,
]
DENSITY_KG_PER_M3 = [
    1232.516351,
    1227.026282,
    1221.118721,
    1214.748899,
    1207.863973,
    1200.401908,
    1192.290048,
    1183.443431,
    1173.762873,
    996.8902499,
]
RPM_VALUES = [270, 290, 310, 330, 350, 370, 390, 410, 430, 450]
SURFACE_TENSION_N_PER_M = 0.0762
EXCLUDE_REASON = "contaminated_source_video_mixing"
CONTAMINATED_TARGET_STEMS = {
    "decay_10fps_visc000.89274_rpm270_renderV",
    "decay_10fps_visc015.97088_rpm330_renderU",
    "decay_10fps_visc021.31029_rpm290_renderR",
    "decay_10fps_visc028.43477_rpm270_renderK",
    "decay_10fps_visc037.94112_rpm270_renderL",
    "decay_10fps_visc067.55088_rpm330_renderW",
    "decay_10fps_visc090.13457_rpm350_renderA",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    source_dir: Path
    render_tags: list[str]
    expected_count: int
    combined_offset: int


def numeric_movs(source_dir: Path) -> list[Path]:
    paths = [
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mov" and not path.name.startswith("._")
    ]
    paths = sorted(paths, key=lambda path: int(path.stem))
    return paths


def format_visc_cp(value: float) -> str:
    integer = int(value)
    fraction = value - integer
    return f"{integer:03d}.{int(round(fraction * 1e5)):05d}"


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open raw video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
    cap.release()
    duration = frame_count / fps if fps > 0.0 else 0.0
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Raw video metadata is invalid for {path}")
    return {
        "source_fps": fps,
        "source_frame_count": frame_count,
        "source_duration_seconds": duration,
        "source_width": width,
        "source_height": height,
    }


def summarize_video_root(video_root: Path) -> dict:
    paths = sorted(video_root.glob("*.mp4"))
    facts = {
        "video_root": str(video_root.relative_to(ROOT)),
        "video_count": len(paths),
        "fps_values": {},
        "frame_count_values": {},
        "width_values": {},
        "height_values": {},
    }
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            continue
        fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 3)
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
        cap.release()
        for key, value in [
            ("fps_values", fps),
            ("frame_count_values", frame_count),
            ("width_values", width),
            ("height_values", height),
        ]:
            value_key = str(value)
            facts[key][value_key] = facts[key].get(value_key, 0) + 1
    return facts


def metadata_for(spec: SourceSpec, group_index: int, source_file: Path) -> tuple[str, dict]:
    block_size = len(RPM_VALUES) * len(spec.render_tags)
    viscosity_block = group_index // block_size
    block_index = group_index % block_size
    rpm_index = block_index % len(RPM_VALUES)
    render_index = block_index // len(RPM_VALUES)

    viscosity_cp = DYNAMIC_VISCOSITY_CP[viscosity_block]
    viscosity_pa_s = viscosity_cp * 1e-3
    density = DENSITY_KG_PER_M3[viscosity_block]
    kinematic_m2_s = viscosity_pa_s / density
    viscosity_str = format_visc_cp(viscosity_cp)
    rpm_value = RPM_VALUES[rpm_index]
    render_tag = spec.render_tags[render_index]
    target_stem = f"decay_10fps_visc{viscosity_str}_rpm{rpm_value}_render{render_tag}"
    use_in_training = target_stem not in CONTAMINATED_TARGET_STEMS

    metadata = {
        "height": 224,
        "width": 224,
        "fps": 10,
        "derived_fps": 10,
        "dynamic_viscosity": viscosity_pa_s,
        "dynamic_viscosity_cP": viscosity_cp,
        "dynamic_viscosity_Pa_s": viscosity_pa_s,
        "dynamic_viscosity_str": viscosity_str,
        "density": density,
        "density_kg_per_m3": density,
        "surface_tension": SURFACE_TENSION_N_PER_M,
        "surface_tension_N_per_m": SURFACE_TENSION_N_PER_M,
        "kinematic_viscosity": kinematic_m2_s,
        "kinematic_viscosity_m2_per_s": kinematic_m2_s,
        "RPM": rpm_value,
        "RPM_index": rpm_index,
        "RENDER": render_tag,
        "INDEX": viscosity_block,
        "viscosity_block_index": viscosity_block,
        "source_group": spec.name,
        "source_file": source_file.name,
        "source_path": str(source_file.relative_to(ROOT)),
        "source_capture_index": int(source_file.stem),
        **probe_video(source_file),
        "group_index": group_index,
        "combined_index": spec.combined_offset + group_index,
        "target_video_stem": target_stem,
        "target_video_file": f"{target_stem}.mp4",
        "usable": use_in_training,
        "use_in_training": use_in_training,
        "exclude_reason": None if use_in_training else EXCLUDE_REASON,
    }
    return target_stem, metadata


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_configs(overwrite: bool, audit_only: bool = False) -> dict:
    specs = [
        SourceSpec(
            name="impeller_1000_originals",
            source_dir=ROOT / "rawdataset/realworld/rawvideos/impeller_1000_originals",
            render_tags=[chr(ord("A") + idx) for idx in range(10)],
            expected_count=1000,
            combined_offset=0,
        ),
        SourceSpec(
            name="raw_real_20rpmincrement_1500",
            source_dir=ROOT / "rawdataset/realworld/rawvideos/raw_real_20rpmincrement_1500",
            render_tags=[chr(ord("K") + idx) for idx in range(15)],
            expected_count=1500,
            combined_offset=1000,
        ),
    ]
    final_root = ROOT / "dataset/RealArchive/real_20rpm_increment_2500"
    consolidated_dir = final_root / "parameters_rebuilt_from_raw"
    manifest_records = []
    report = {
        "consolidated_config_dir": str(consolidated_dir.relative_to(ROOT)),
        "source_groups": [],
        "expected_total": sum(spec.expected_count for spec in specs),
        "written_total": 0,
    }

    for spec in specs:
        files = numeric_movs(spec.source_dir)
        if len(files) != spec.expected_count:
            raise RuntimeError(f"{spec.source_dir} expected {spec.expected_count} MOVs, found {len(files)}")
        group_dir = spec.source_dir / "configs"
        group_written = 0
        for group_index, source_file in enumerate(files):
            target_stem, metadata = metadata_for(spec, group_index, source_file)
            group_path = group_dir / f"{target_stem}.json"
            final_path = consolidated_dir / f"{target_stem}.json"
            if not audit_only and not overwrite and (group_path.exists() or final_path.exists()):
                raise FileExistsError(f"Refusing to overwrite existing config: {group_path} or {final_path}")
            if not audit_only:
                write_json(group_path, metadata)
                write_json(final_path, metadata)
            manifest_records.append(metadata)
            group_written += 1

        report["source_groups"].append(
            {
                "name": spec.name,
                "source_dir": str(spec.source_dir.relative_to(ROOT)),
                "source_count": len(files),
                "render_tags": spec.render_tags,
                "config_dir": str(group_dir.relative_to(ROOT)),
                "written": group_written,
            }
        )
        report["written_total"] += group_written

    if not audit_only:
        write_json(final_root / "raw_config_manifest.json", manifest_records)

    existing_videos = {path.stem for path in (final_root / "videos").glob("*.mp4")}
    all_rebuilt_configs = {record["target_video_stem"] for record in manifest_records}
    usable_rebuilt_configs = {record["target_video_stem"] for record in manifest_records if record["use_in_training"]}
    excluded_records = [record for record in manifest_records if not record["use_in_training"]]
    existing_parameters = {path.stem for path in (final_root / "parameters").glob("*.json")}
    report.update(
        {
            "manifest_path": str((final_root / "raw_config_manifest.json").relative_to(ROOT)),
            "audit_only": audit_only,
            "excluded_config_count": len(excluded_records),
            "excluded_raw_sources": [
                {
                    "target_video_stem": record["target_video_stem"],
                    "source_group": record["source_group"],
                    "source_file": record["source_file"],
                    "source_path": record["source_path"],
                    "exclude_reason": record["exclude_reason"],
                }
                for record in excluded_records
            ],
            "existing_final_video_count": len(existing_videos),
            "existing_final_parameter_count": len(existing_parameters),
            "missing_existing_videos_for_rebuilt_configs": sorted(all_rebuilt_configs - existing_videos),
            "missing_existing_parameters_for_rebuilt_configs": sorted(all_rebuilt_configs - existing_parameters),
            "missing_existing_videos_for_active_rebuilt_configs": sorted(usable_rebuilt_configs - existing_videos),
            "missing_existing_parameters_for_active_rebuilt_configs": sorted(usable_rebuilt_configs - existing_parameters),
            "existing_videos_not_in_rebuilt_configs": sorted(existing_videos - all_rebuilt_configs),
            "existing_parameters_not_in_rebuilt_configs": sorted(existing_parameters - all_rebuilt_configs),
            "existing_generated_dataset_facts": {
                "real_20rpm_increment_2500": summarize_video_root(final_root / "videos"),
                "train_993_wo_pat2": summarize_video_root(ROOT / "dataset/RealArchive/train_993_wo_pat2/videos"),
                "test_1000_wo_pat2": summarize_video_root(ROOT / "dataset/RealArchive/test_1000_wo_pat2/videos"),
            },
        }
    )
    if not audit_only:
        write_json(final_root / "raw_config_build_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    report = build_configs(overwrite=args.overwrite, audit_only=args.audit_only)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
