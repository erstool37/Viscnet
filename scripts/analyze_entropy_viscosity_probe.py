#!/usr/bin/env python3
"""CPU-only entropy probe for real viscosity videos."""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", message="An input array is constant.*")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="The 'labels' parameter of boxplot.*")

FEATURE_COLUMNS = [
    "pattern_entropy",
    "pattern_grad_entropy",
    "pattern_spectral_entropy",
    "video_entropy_mean",
    "video_entropy_std",
    "video_grad_entropy_mean",
    "video_grad_entropy_std",
    "video_spectral_entropy_mean",
    "video_spectral_entropy_std",
    "residual_entropy_mean",
    "residual_entropy_std",
    "temporal_diff_entropy_mean",
    "temporal_diff_entropy_std",
    "pattern_video_mi_mean",
    "pattern_video_mi_std",
    "pattern_video_corr_mean",
    "pattern_video_corr_std",
    "entropy_delta_mean",
    "grad_entropy_delta_mean",
    "spectral_entropy_delta_mean",
    "residual_minus_pattern_entropy",
    "temporal_minus_pattern_entropy",
]


def finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def sortable_abs(value: float | None) -> float:
    return abs(value) if value is not None else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset/RealArchive/test_1000_wo_pat2")
    parser.add_argument(
        "--prediction-root",
        default=(
            "outputs/rebuild_reproduction/repro_realonly_993_window30x21_no_rpm_ep50_1000x21_inference_rerun/"
            "repro_realonly_993_window30x21_no_rpm_ep50_realtest_1000x21_windows_rerun"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000",
    )
    parser.add_argument("--frame-count", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-videos", type=int, default=None)
    return parser.parse_args()


def shannon_entropy_uint8(image: np.ndarray) -> float:
    values = np.asarray(image, dtype=np.uint8).ravel()
    hist = np.bincount(values, minlength=256).astype(np.float64)
    prob = hist[hist > 0] / values.size
    return float(-(prob * np.log2(prob)).sum())


def gradient_entropy(gray: np.ndarray) -> float:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    if float(magnitude.max()) <= 0.0:
        quantized = np.zeros_like(gray, dtype=np.uint8)
    else:
        quantized = np.clip(magnitude / magnitude.max() * 255.0, 0, 255).astype(np.uint8)
    return shannon_entropy_uint8(quantized)


def spectral_entropy(gray: np.ndarray) -> float:
    arr = gray.astype(np.float32)
    arr = arr - float(arr.mean())
    power = np.abs(np.fft.rfft2(arr)) ** 2
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    prob = (power / total).ravel()
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum() / math.log2(prob.size))


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    hist_2d, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins, range=[[0, 255], [0, 255]])
    total = hist_2d.sum()
    if total <= 0:
        return 0.0
    pxy = hist_2d / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float((pxy[nz] * np.log2(pxy[nz] / px_py[nz])).sum())


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64).ravel()
    bb = b.astype(np.float64).ravel()
    if float(aa.std()) == 0.0 or float(bb.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def center_crop_or_resize(gray: np.ndarray, size: int) -> np.ndarray:
    height, width = gray.shape[:2]
    if height >= size and width >= size:
        top = (height - size) // 2
        left = (width - size) // 2
        return gray[top : top + size, left : left + size]
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def load_pattern(dataset_root: Path, background_name: str, image_size: int) -> np.ndarray:
    path = dataset_root / "backgrounds" / f"{background_name}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return center_crop_or_resize(image, image_size)


def load_video_frames(path: Path, frame_count: int, image_size: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(gray)
    cap.release()
    if len(frames) < frame_count:
        raise ValueError(f"{path} has only {len(frames)} readable frames; expected {frame_count}")
    return frames


def load_params(path: Path) -> dict:
    with path.open("r") as file:
        return json.load(file)


def prediction_lookup(prediction_root: Path) -> tuple[dict[str, dict], dict[tuple[str, int], dict], dict]:
    with (prediction_root / "variable_window_predictions_mean_logits_per_video.json").open("r") as file:
        video_predictions = json.load(file)
    with (prediction_root / "variable_window_predictions_per_window.json").open("r") as file:
        window_predictions = json.load(file)
    with (prediction_root / "summary.json").open("r") as file:
        summary = json.load(file)
    by_name = {record["name"]: record for record in video_predictions}
    by_name_start = {(record["name"], int(record["window_start"])): record for record in window_predictions}
    return by_name, by_name_start, summary


def natural_video_paths(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "videos").glob("*.mp4"), key=lambda path: path.name)


def aggregate_window(values: list[float], start: int, size: int) -> tuple[float, float]:
    arr = np.asarray(values[start : start + size], dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def video_entropy_records(
    video_path: Path,
    dataset_root: Path,
    video_pred: dict,
    window_preds: dict[tuple[str, int], dict],
    frame_count: int,
    window_size: int,
    image_size: int,
) -> tuple[dict, list[dict]]:
    name = video_path.stem
    params = load_params(dataset_root / "parametersNorm" / f"{name}.json")
    background_name = str(params.get("background", infer_background_from_name(name)))
    pattern = load_pattern(dataset_root, background_name, image_size)
    frames = load_video_frames(video_path, frame_count, image_size)

    pattern_entropy = shannon_entropy_uint8(pattern)
    pattern_grad_entropy = gradient_entropy(pattern)
    pattern_spectral_entropy = spectral_entropy(pattern)

    frame_entropy = []
    frame_grad_entropy = []
    frame_spectral_entropy = []
    residual_entropy = []
    mi = []
    corr = []
    for frame in frames:
        frame_entropy.append(shannon_entropy_uint8(frame))
        frame_grad_entropy.append(gradient_entropy(frame))
        frame_spectral_entropy.append(spectral_entropy(frame))
        residual = cv2.absdiff(frame, pattern)
        residual_entropy.append(shannon_entropy_uint8(residual))
        mi.append(mutual_information(pattern, frame))
        corr.append(safe_corr(pattern, frame))

    temporal_diff_entropy = [
        shannon_entropy_uint8(cv2.absdiff(frames[idx], frames[idx - 1])) for idx in range(1, len(frames))
    ]

    def mean_std(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    video_entropy_mean, video_entropy_std = mean_std(frame_entropy)
    video_grad_entropy_mean, video_grad_entropy_std = mean_std(frame_grad_entropy)
    video_spectral_entropy_mean, video_spectral_entropy_std = mean_std(frame_spectral_entropy)
    residual_entropy_mean, residual_entropy_std = mean_std(residual_entropy)
    temporal_diff_entropy_mean, temporal_diff_entropy_std = mean_std(temporal_diff_entropy)
    mi_mean, mi_std = mean_std(mi)
    corr_mean, corr_std = mean_std(corr)

    base = {
        "name": name,
        "background": int(background_name),
        "true_viscosity_class": int(video_pred["true_viscosity_class"]),
        "prediction": int(video_pred["prediction"]),
        "correct": bool(video_pred["correct"]),
        "rpm_value": parse_rpm_from_name(name),
        "viscosity_value": parse_viscosity_from_name(name),
        "pattern_entropy": pattern_entropy,
        "pattern_grad_entropy": pattern_grad_entropy,
        "pattern_spectral_entropy": pattern_spectral_entropy,
        "video_entropy_mean": video_entropy_mean,
        "video_entropy_std": video_entropy_std,
        "video_grad_entropy_mean": video_grad_entropy_mean,
        "video_grad_entropy_std": video_grad_entropy_std,
        "video_spectral_entropy_mean": video_spectral_entropy_mean,
        "video_spectral_entropy_std": video_spectral_entropy_std,
        "residual_entropy_mean": residual_entropy_mean,
        "residual_entropy_std": residual_entropy_std,
        "temporal_diff_entropy_mean": temporal_diff_entropy_mean,
        "temporal_diff_entropy_std": temporal_diff_entropy_std,
        "pattern_video_mi_mean": mi_mean,
        "pattern_video_mi_std": mi_std,
        "pattern_video_corr_mean": corr_mean,
        "pattern_video_corr_std": corr_std,
        "entropy_delta_mean": video_entropy_mean - pattern_entropy,
        "grad_entropy_delta_mean": video_grad_entropy_mean - pattern_grad_entropy,
        "spectral_entropy_delta_mean": video_spectral_entropy_mean - pattern_spectral_entropy,
        "residual_minus_pattern_entropy": residual_entropy_mean - pattern_entropy,
        "temporal_minus_pattern_entropy": temporal_diff_entropy_mean - pattern_entropy,
    }

    window_records = []
    for start in range(0, frame_count - window_size + 1):
        pred = window_preds[(name, start)]
        window_frame_entropy_mean, window_frame_entropy_std = aggregate_window(frame_entropy, start, window_size)
        window_grad_entropy_mean, window_grad_entropy_std = aggregate_window(frame_grad_entropy, start, window_size)
        window_spectral_entropy_mean, window_spectral_entropy_std = aggregate_window(
            frame_spectral_entropy, start, window_size
        )
        window_residual_entropy_mean, window_residual_entropy_std = aggregate_window(
            residual_entropy, start, window_size
        )
        window_mi_mean, window_mi_std = aggregate_window(mi, start, window_size)
        window_corr_mean, window_corr_std = aggregate_window(corr, start, window_size)
        temporal_slice = temporal_diff_entropy[start : start + window_size - 1]
        window_temporal_mean, window_temporal_std = mean_std(temporal_slice)
        window_records.append(
            {
                "name": name,
                "window_start": start,
                "background": int(background_name),
                "true_viscosity_class": int(pred["true_viscosity_class"]),
                "prediction": int(pred["prediction"]),
                "correct": bool(pred["correct"]),
                "rpm_value": float(pred.get("rpm_value", base["rpm_value"])),
                "viscosity_value": float(pred.get("viscosity_value", base["viscosity_value"])),
                "pattern_entropy": pattern_entropy,
                "pattern_grad_entropy": pattern_grad_entropy,
                "pattern_spectral_entropy": pattern_spectral_entropy,
                "video_entropy_mean": window_frame_entropy_mean,
                "video_entropy_std": window_frame_entropy_std,
                "video_grad_entropy_mean": window_grad_entropy_mean,
                "video_grad_entropy_std": window_grad_entropy_std,
                "video_spectral_entropy_mean": window_spectral_entropy_mean,
                "video_spectral_entropy_std": window_spectral_entropy_std,
                "residual_entropy_mean": window_residual_entropy_mean,
                "residual_entropy_std": window_residual_entropy_std,
                "temporal_diff_entropy_mean": window_temporal_mean,
                "temporal_diff_entropy_std": window_temporal_std,
                "pattern_video_mi_mean": window_mi_mean,
                "pattern_video_mi_std": window_mi_std,
                "pattern_video_corr_mean": window_corr_mean,
                "pattern_video_corr_std": window_corr_std,
                "entropy_delta_mean": window_frame_entropy_mean - pattern_entropy,
                "grad_entropy_delta_mean": window_grad_entropy_mean - pattern_grad_entropy,
                "spectral_entropy_delta_mean": window_spectral_entropy_mean - pattern_spectral_entropy,
                "residual_minus_pattern_entropy": window_residual_entropy_mean - pattern_entropy,
                "temporal_minus_pattern_entropy": window_temporal_mean - pattern_entropy,
            }
        )
    return base, window_records


def infer_background_from_name(name: str) -> int:
    render = name.split("_render")[-1]
    if render in set("ABCDEFGHIJ"):
        return 1
    if render in set("KLMNO"):
        return 2
    if render in set("PQRST"):
        return 3
    if render in set("UVWXY"):
        return 4
    return 1


def parse_rpm_from_name(name: str) -> float:
    match = re.search(r"_rpm([0-9.]+)", name)
    return float(match.group(1)) if match else float("nan")


def parse_viscosity_from_name(name: str) -> float:
    match = re.search(r"_visc([0-9.]+)_", name)
    return float(match.group(1)) if match else float("nan")


def summarize_feature_table(df: pd.DataFrame) -> dict:
    correlations = {}
    for column in FEATURE_COLUMNS:
        spearman = finite_or_none(df[column].corr(df["true_viscosity_class"], method="spearman"))
        pearson = finite_or_none(df[column].corr(df["true_viscosity_class"], method="pearson"))
        correlations[column] = {
            "spearman_true_class": spearman,
            "pearson_true_class": pearson,
            "mean_correct": finite_or_none(df.loc[df["correct"], column].mean()),
            "mean_wrong": finite_or_none(df.loc[~df["correct"], column].mean()) if (~df["correct"]).any() else None,
        }
    by_background = {}
    for background, group in df.groupby("background"):
        by_background[str(int(background))] = {
            column: finite_or_none(group[column].corr(group["true_viscosity_class"], method="spearman"))
            for column in FEATURE_COLUMNS
        }
    by_rpm = {}
    for rpm, group in df.groupby("rpm_value"):
        if len(group["true_viscosity_class"].unique()) < 2:
            continue
        by_rpm[str(int(rpm))] = {
            column: finite_or_none(group[column].corr(group["true_viscosity_class"], method="spearman"))
            for column in FEATURE_COLUMNS
        }
    return {
        "rows": int(len(df)),
        "correct_rows": int(df["correct"].sum()),
        "wrong_rows": int((~df["correct"]).sum()),
        "model_accuracy_in_joined_table": float(df["correct"].mean()),
        "feature_correlations": correlations,
        "within_background_spearman": by_background,
        "within_rpm_spearman": by_rpm,
    }


def entropy_baselines(df: pd.DataFrame, output_dir: Path, prefix: str) -> dict:
    x = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = df["true_viscosity_class"].astype(int).to_numpy()
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2 or int(counts.min()) < 2:
        return {
            "skipped": {
                "reason": "Need at least two classes and at least two samples per class for stratified CV.",
                "labels": [int(label) for label in labels],
                "counts": [int(count) for count in counts],
            }
        }
    n_splits = min(5, int(counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1205)
    baselines = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=1205,
            class_weight="balanced",
            n_jobs=1,
            max_depth=8,
        ),
    }
    results = {}
    for name, model in baselines.items():
        pred = cross_val_predict(model, x, y, cv=cv, n_jobs=1)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred, labels=list(range(10)))
        results[name] = {
            "cv_accuracy": float(acc),
            "confusion_matrix_counts": cm.astype(int).tolist(),
            "prediction_counts": {str(idx): int((pred == idx).sum()) for idx in range(10)},
        }
        save_confusion_plot(cm, output_dir / f"{prefix}_{name}_confusion.png", f"{prefix} {name} entropy-only")
    return results


def save_confusion_plot(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_feature_by_class(df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    columns = [
        "pattern_entropy",
        "video_entropy_mean",
        "entropy_delta_mean",
        "residual_entropy_mean",
        "temporal_diff_entropy_mean",
        "pattern_video_mi_mean",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, column in zip(axes.ravel(), columns):
        grouped = [df.loc[df["true_viscosity_class"] == cls, column].dropna().to_numpy() for cls in range(10)]
        ax.boxplot(grouped, tick_labels=list(range(10)), showfliers=False)
        ax.set_title(column)
        ax.set_xlabel("true viscosity class")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_entropy_by_true_class.png", dpi=250)
    plt.close(fig)


def plot_feature_correlation_bars(summary: dict, output_dir: Path, prefix: str) -> None:
    items = [
        (column, sortable_abs(values["spearman_true_class"]))
        for column, values in summary["feature_correlations"].items()
    ]
    items.sort(key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in items[:12]]
    values = [item[1] for item in items[:12]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlabel("|Spearman correlation with true viscosity class|")
    ax.set_title(f"{prefix} entropy feature correlations")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_feature_correlations.png", dpi=250)
    plt.close(fig)


def plot_correct_wrong(df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    columns = ["entropy_delta_mean", "residual_entropy_mean", "temporal_diff_entropy_mean", "pattern_video_mi_mean"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, column in zip(axes, columns):
        groups = [df.loc[df["correct"], column].dropna().to_numpy(), df.loc[~df["correct"], column].dropna().to_numpy()]
        ax.boxplot(groups, tick_labels=["correct", "wrong"], showfliers=False)
        ax.set_title(column)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_correct_wrong_entropy.png", dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    prediction_root = Path(args.prediction_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_preds, window_preds, model_summary = prediction_lookup(prediction_root)
    video_paths = natural_video_paths(dataset_root)
    if args.max_videos is not None:
        video_paths = video_paths[: args.max_videos]

    video_records = []
    window_records = []
    for idx, video_path in enumerate(video_paths, start=1):
        name = video_path.stem
        if name not in video_preds:
            raise KeyError(f"Missing video prediction for {name}")
        video_record, sample_windows = video_entropy_records(
            video_path=video_path,
            dataset_root=dataset_root,
            video_pred=video_preds[name],
            window_preds=window_preds,
            frame_count=args.frame_count,
            window_size=args.window_size,
            image_size=args.image_size,
        )
        video_records.append(video_record)
        window_records.extend(sample_windows)
        if idx % 50 == 0:
            print(f"processed {idx}/{len(video_paths)} videos", flush=True)

    video_df = pd.DataFrame(video_records)
    window_df = pd.DataFrame(window_records)
    video_csv = output_dir / "video_entropy_features.csv"
    window_csv = output_dir / "window_entropy_features.csv"
    video_df.to_csv(video_csv, index=False)
    window_df.to_csv(window_csv, index=False)

    video_summary = summarize_feature_table(video_df)
    window_summary = summarize_feature_table(window_df)
    video_baselines = entropy_baselines(video_df, output_dir, "video_level")
    window_sample_df = window_df.groupby("name", as_index=False).first()
    window_video_baselines = entropy_baselines(window_sample_df, output_dir, "window_start0_video_level")

    plot_feature_by_class(video_df, output_dir, "video_level")
    plot_feature_correlation_bars(video_summary, output_dir, "video_level")
    plot_correct_wrong(video_df, output_dir, "video_level")
    plot_feature_by_class(window_df, output_dir, "window_level")
    plot_feature_correlation_bars(window_summary, output_dir, "window_level")
    plot_correct_wrong(window_df, output_dir, "window_level")

    top_video = sorted(
        video_summary["feature_correlations"].items(),
        key=lambda item: sortable_abs(item[1]["spearman_true_class"]),
        reverse=True,
    )[:10]
    top_window = sorted(
        window_summary["feature_correlations"].items(),
        key=lambda item: sortable_abs(item[1]["spearman_true_class"]),
        reverse=True,
    )[:10]

    summary = {
        "dataset_root": str(dataset_root),
        "prediction_root": str(prediction_root),
        "output_dir": str(output_dir),
        "cpu_only": True,
        "video_count": int(len(video_df)),
        "window_count": int(len(window_df)),
        "frame_count": args.frame_count,
        "window_size": args.window_size,
        "model_reference": {
            "run_name": model_summary.get("run_name"),
            "checkpoint": model_summary.get("checkpoint"),
            "rpm_bool": model_summary.get("rpm_bool"),
            "runtime_zeroed_rpm_confirmed": model_summary.get("runtime_zeroed_rpm_confirmed"),
            "per_window_accuracy": model_summary["variable_window_all_starts_per_window"]["accuracy"],
            "per_window_confusion_matrix": model_summary["variable_window_all_starts_per_window"][
                "confusion_matrix_path"
            ],
            "mean_logits_per_video_accuracy": model_summary["variable_window_mean_logits_per_video"]["accuracy"],
            "mean_logits_per_video_confusion_matrix": model_summary["variable_window_mean_logits_per_video"][
                "confusion_matrix_path"
            ],
        },
        "video_level_entropy_summary": video_summary,
        "window_level_entropy_summary": window_summary,
        "entropy_only_video_level_baselines": video_baselines,
        "entropy_only_window_start0_video_level_baselines": window_video_baselines,
        "top_video_level_correlations": [
            {"feature": name, **values} for name, values in top_video
        ],
        "top_window_level_correlations": [
            {"feature": name, **values} for name, values in top_window
        ],
        "artifacts": {
            "video_entropy_features_csv": str(video_csv),
            "window_entropy_features_csv": str(window_csv),
            "video_level_entropy_by_true_class": str(output_dir / "video_level_entropy_by_true_class.png"),
            "video_level_feature_correlations": str(output_dir / "video_level_feature_correlations.png"),
            "video_level_correct_wrong_entropy": str(output_dir / "video_level_correct_wrong_entropy.png"),
            "window_level_entropy_by_true_class": str(output_dir / "window_level_entropy_by_true_class.png"),
            "window_level_feature_correlations": str(output_dir / "window_level_feature_correlations.png"),
            "window_level_correct_wrong_entropy": str(output_dir / "window_level_correct_wrong_entropy.png"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown_report(summary, output_dir / "report.md")
    print(summary_path)


def write_markdown_report(summary: dict, path: Path) -> None:
    def fmt(value: float | None) -> str:
        return "null" if value is None else f"{value:.4f}"

    lines = [
        "# Entropy Probe: No-RPM Window-Sliding Real Test",
        "",
        "## Reference Model",
        "",
        f"- Run: `{summary['model_reference']['run_name']}`",
        f"- Checkpoint: `{summary['model_reference']['checkpoint']}`",
        f"- Runtime zeroed RPM confirmed: `{summary['model_reference']['runtime_zeroed_rpm_confirmed']}`",
        f"- Per-window accuracy: `{summary['model_reference']['per_window_accuracy']}`",
        f"- Mean-logits per-video accuracy: `{summary['model_reference']['mean_logits_per_video_accuracy']}`",
        "",
        "## Entropy-Only Baselines",
        "",
    ]
    for name, result in summary["entropy_only_video_level_baselines"].items():
        if "cv_accuracy" in result:
            lines.append(f"- Video-level `{name}` CV accuracy: `{result['cv_accuracy']}`")
        else:
            lines.append(f"- Video-level `{name}`: `{result}`")
    lines.append("")
    lines.append("## Top Video-Level Correlations")
    lines.append("")
    for row in summary["top_video_level_correlations"]:
        lines.append(
            f"- `{row['feature']}`: Spearman `{fmt(row['spearman_true_class'])}`, "
            f"Pearson `{fmt(row['pearson_true_class'])}`"
        )
    lines.append("")
    lines.append("## Top Window-Level Correlations")
    lines.append("")
    for row in summary["top_window_level_correlations"]:
        lines.append(
            f"- `{row['feature']}`: Spearman `{fmt(row['spearman_true_class'])}`, "
            f"Pearson `{fmt(row['pearson_true_class'])}`"
        )
    lines.append("")
    lines.append("## Interpretation Guardrail")
    lines.append("")
    lines.append(
        "This probe tests whether simple entropy summaries explain the high no-RPM "
        "window-sliding result. Entropy correlations or entropy-only baselines are "
        "diagnostic evidence, not proof of causal viscosity reasoning."
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
