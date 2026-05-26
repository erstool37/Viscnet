#!/usr/bin/env python3
"""Build fixed-length temporal-window clips from real Viscnet videos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_records(manifest: Path) -> list[dict]:
    payload = json.loads(manifest.read_text())
    records = payload["samples"] if isinstance(payload, dict) else payload
    return list(records)


def read_video(path: Path) -> tuple[list, float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    width = height = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        frames.append(frame)
    cap.release()
    return frames, fps, (width, height)


def write_clip(path: Path, frames: list, fps: float, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def parse_phase_offsets(value: str, sample_stride: int) -> list[int]:
    if sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")
    if value == "all":
        return list(range(sample_stride))
    offsets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not offsets:
        raise ValueError("--phase-offsets must provide at least one offset")
    invalid = [offset for offset in offsets if offset < 0 or offset >= sample_stride]
    if invalid:
        raise ValueError(f"phase offsets must be in [0, {sample_stride - 1}], got {invalid}")
    return offsets


def sampled_indices(start: int, window_size: int, sample_stride: int) -> list[int]:
    return [start + idx * sample_stride for idx in range(window_size)]


def ensure_backgrounds(records: list[dict], output_root: Path) -> None:
    source_video = ROOT / records[0]["video_path"]
    source_backgrounds = source_video.parents[1] / "backgrounds"
    target_backgrounds = output_root / "backgrounds"
    if target_backgrounds.exists() or target_backgrounds.is_symlink():
        return
    try:
        os.symlink(os.path.relpath(source_backgrounds, output_root), target_backgrounds, target_is_directory=True)
    except OSError:
        shutil.copytree(source_backgrounds, target_backgrounds)


def existing_manifest_ok(manifest: Path, expected_count: int) -> bool:
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return False
    samples = payload.get("samples", [])
    if len(samples) != expected_count:
        return False
    return all((ROOT / sample["video_path"]).exists() for sample in samples[:20])


def build(args: argparse.Namespace) -> None:
    source_manifest = ROOT / args.source_manifest
    output_root = ROOT / args.output_root
    output_videos = output_root / "videos"
    output_manifest = output_root / "manifest.json"
    output_summary = output_root / "summary.json"
    records = load_records(source_manifest)
    if args.max_records is not None:
        records = records[: max(1, args.max_records)]
    phase_offsets = parse_phase_offsets(args.phase_offsets, args.sample_stride)
    expected_count = len(records) * args.windows_per_video * len(phase_offsets)

    if not args.force and existing_manifest_ok(output_manifest, expected_count):
        print(f"Window dataset already exists: {relative(output_manifest)} ({expected_count} samples)")
        return

    output_videos.mkdir(parents=True, exist_ok=True)
    ensure_backgrounds(records, output_root)

    samples = []
    skipped_windows = []
    for record_index, record in enumerate(records):
        source_video = ROOT / record["video_path"]
        frames, fps, size = read_video(source_video)
        required_frames = args.windows_per_video + (args.window_size - 1) * args.sample_stride
        if len(frames) < required_frames and not args.allow_partial_phase_offsets:
            raise ValueError(
                f"{source_video} has {len(frames)} frames; need at least "
                f"{required_frames} for window_size={args.window_size}, "
                f"windows_per_video={args.windows_per_video}, sample_stride={args.sample_stride}"
            )
        for start in range(args.windows_per_video):
            for phase_offset in phase_offsets:
                effective_start = start + phase_offset
                indices = sampled_indices(effective_start, args.window_size, args.sample_stride)
                if indices[-1] >= len(frames):
                    skipped_windows.append(
                        {
                            "source_video_path": record["video_path"],
                            "window_start": start,
                            "phase_offset": phase_offset,
                            "last_required_frame": indices[-1],
                            "source_frame_count": len(frames),
                        }
                    )
                    continue
                clip_frames = [frames[index] for index in indices]
                if args.sample_stride == 1 and phase_offset == 0:
                    clip_name = f"{source_video.stem}_w{start:02d}.mp4"
                else:
                    clip_name = f"{source_video.stem}_w{start:02d}_s{args.sample_stride}_p{phase_offset}.mp4"
                clip_path = output_videos / clip_name
                if args.force or not clip_path.exists():
                    write_clip(clip_path, clip_frames, fps / args.sample_stride, size)
                sample = dict(record)
                sample["video_path"] = relative(clip_path)
                sample["parameters_norm_path"] = record["parameters_norm_path"]
                sample["source_video_path"] = record["video_path"]
                sample["source_index"] = record_index
                sample["window_start"] = start
                sample["window_size"] = args.window_size
                sample["sample_stride"] = args.sample_stride
                sample["phase_offset"] = phase_offset
                sample["source_frame_indices"] = indices
                sample["source_fps"] = fps
                sample["derived_fps"] = fps / args.sample_stride
                samples.append(sample)

    payload = {
        "seed": args.seed,
        "source_manifest": args.source_manifest,
        "window_size": args.window_size,
        "windows_per_video": args.windows_per_video,
        "sample_stride": args.sample_stride,
        "phase_offsets": phase_offsets,
        "sample_count": len(samples),
        "skipped_window_count": len(skipped_windows),
        "skipped_windows": skipped_windows[:100],
        "samples": samples,
    }
    output_manifest.write_text(json.dumps(payload, indent=2) + "\n")
    output_summary.write_text(
        json.dumps(
            {
                "source_manifest": args.source_manifest,
                "output_manifest": relative(output_manifest),
                "source_count": len(records),
                "sample_count": len(samples),
                "window_size": args.window_size,
                "windows_per_video": args.windows_per_video,
                "sample_stride": args.sample_stride,
                "phase_offsets": phase_offsets,
                "skipped_window_count": len(skipped_windows),
                "skipped_windows_preview": skipped_windows[:20],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(samples)} window clips -> {relative(output_manifest)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", default="configs/rebuild/manifests/real_train_993.json")
    parser.add_argument(
        "--output-root",
        default="outputs/rebuild_reproduction/derived_datasets/real_train_993_windows30_stride1",
    )
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--windows-per-video", type=int, default=21)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument(
        "--phase-offsets",
        default="0",
        help='Comma-separated offsets, or "all" for offsets 0..sample_stride-1.',
    )
    parser.add_argument(
        "--allow-partial-phase-offsets",
        action="store_true",
        help="Skip invalid phase/window combinations instead of requiring every requested offset to fit.",
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
