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
    expected_count = len(records) * args.windows_per_video

    if not args.force and existing_manifest_ok(output_manifest, expected_count):
        print(f"Window dataset already exists: {relative(output_manifest)} ({expected_count} samples)")
        return

    output_videos.mkdir(parents=True, exist_ok=True)
    ensure_backgrounds(records, output_root)

    samples = []
    for record_index, record in enumerate(records):
        source_video = ROOT / record["video_path"]
        frames, fps, size = read_video(source_video)
        if len(frames) < args.window_size + args.windows_per_video - 1:
            raise ValueError(
                f"{source_video} has {len(frames)} frames; need at least "
                f"{args.window_size + args.windows_per_video - 1}"
            )
        for start in range(args.windows_per_video):
            clip_frames = frames[start : start + args.window_size]
            clip_name = f"{source_video.stem}_w{start:02d}.mp4"
            clip_path = output_videos / clip_name
            if args.force or not clip_path.exists():
                write_clip(clip_path, clip_frames, fps, size)
            sample = dict(record)
            sample["video_path"] = relative(clip_path)
            sample["parameters_norm_path"] = record["parameters_norm_path"]
            sample["source_video_path"] = record["video_path"]
            sample["source_index"] = record_index
            sample["window_start"] = start
            sample["window_size"] = args.window_size
            samples.append(sample)

    payload = {
        "seed": args.seed,
        "source_manifest": args.source_manifest,
        "window_size": args.window_size,
        "windows_per_video": args.windows_per_video,
        "sample_count": len(samples),
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
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
