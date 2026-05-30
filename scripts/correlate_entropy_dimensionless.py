#!/usr/bin/env python3
"""Correlate entropy-change features with Re/Ca proxies and model correctness."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", message="An input array is constant.*")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


ENTROPY_CHANGE_COLUMNS = [
    "entropy_delta_mean",
    "grad_entropy_delta_mean",
    "spectral_entropy_delta_mean",
    "residual_minus_pattern_entropy",
    "temporal_minus_pattern_entropy",
    "residual_entropy_mean",
    "temporal_diff_entropy_mean",
    "temporal_diff_entropy_std",
    "pattern_video_mi_mean",
    "pattern_video_corr_mean",
]

PHYSICS_COLUMNS = [
    "Re_proxy",
    "log10_Re_proxy",
    "Ca_proxy",
    "log10_Ca_proxy",
    "We_proxy",
    "log10_We_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entropy-dir",
        default="outputs/rebuild_reproduction/entropy_probe/repro_realonly_993_window30x21_no_rpm_ep50_test1000",
    )
    parser.add_argument("--dataset-root", default="dataset/RealArchive/test_1000_wo_pat2")
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/rebuild_reproduction/entropy_probe/"
            "repro_realonly_993_window30x21_no_rpm_ep50_test1000_dimensionless"
        ),
    )
    return parser.parse_args()


def finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def load_raw_params(dataset_root: Path) -> dict[str, dict]:
    records = {}
    for path in sorted((dataset_root / "parameters").glob("*.json")):
        with path.open("r") as file:
            data = json.load(file)
        name = path.stem
        rho = float(data["density"])
        nu = float(data["kinematic_viscosity"])
        sigma = float(data["surface_tension"])
        rpm = float(data.get("RPM", data.get("rpm")))
        omega = 2.0 * math.pi * rpm / 60.0
        mu_from_nu = rho * nu
        records[name] = {
            "density_raw": rho,
            "kinematic_viscosity_raw": nu,
            "dynamic_viscosity_from_rho_nu": mu_from_nu,
            "surface_tension_raw": sigma,
            "rpm_raw": rpm,
            "omega_rad_s": omega,
            "Re_proxy": omega / nu,
            "Ca_proxy": mu_from_nu * omega / sigma,
            "We_proxy": rho * omega * omega / sigma,
        }
    return records


def attach_physics(df: pd.DataFrame, raw_params: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name in df["name"]:
        if name not in raw_params:
            raise KeyError(f"Missing raw parameters for {name}")
        rows.append(raw_params[name])
    physics = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), physics.reset_index(drop=True)], axis=1)
    out["log10_Re_proxy"] = np.log10(out["Re_proxy"])
    out["log10_Ca_proxy"] = np.log10(out["Ca_proxy"])
    out["log10_We_proxy"] = np.log10(out["We_proxy"])
    out["correct_int"] = out["correct"].astype(int)
    return out


def corr(df: pd.DataFrame, left: str, right: str, method: str) -> float | None:
    return finite_or_none(df[left].corr(df[right], method=method))


def correlation_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    for physics in PHYSICS_COLUMNS:
        for entropy in ENTROPY_CHANGE_COLUMNS:
            rows.append(
                {
                    "physics_feature": physics,
                    "entropy_feature": entropy,
                    "pearson": corr(df, physics, entropy, "pearson"),
                    "spearman": corr(df, physics, entropy, "spearman"),
                }
            )
    rows.sort(key=lambda row: abs(row["spearman"] or 0.0), reverse=True)
    return rows


def correctness_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    for feature in PHYSICS_COLUMNS + ENTROPY_CHANGE_COLUMNS:
        correct = df.loc[df["correct"], feature]
        wrong = df.loc[~df["correct"], feature]
        rows.append(
            {
                "feature": feature,
                "pearson_with_correct": corr(df, feature, "correct_int", "pearson"),
                "spearman_with_correct": corr(df, feature, "correct_int", "spearman"),
                "mean_correct": finite_or_none(correct.mean()),
                "mean_wrong": finite_or_none(wrong.mean()) if len(wrong) else None,
                "std_correct": finite_or_none(correct.std()),
                "std_wrong": finite_or_none(wrong.std()) if len(wrong) else None,
            }
        )
    rows.sort(key=lambda row: abs(row["pearson_with_correct"] or 0.0), reverse=True)
    return rows


def grouped_accuracy(df: pd.DataFrame, by: str) -> list[dict]:
    rows = []
    for value, group in df.groupby(by):
        rows.append(
            {
                by: int(value) if float(value).is_integer() else float(value),
                "count": int(len(group)),
                "accuracy": float(group["correct_int"].mean()),
                "mean_log10_Re_proxy": float(group["log10_Re_proxy"].mean()),
                "mean_log10_Ca_proxy": float(group["log10_Ca_proxy"].mean()),
                "mean_temporal_diff_entropy": float(group["temporal_diff_entropy_mean"].mean()),
                "mean_entropy_delta": float(group["entropy_delta_mean"].mean()),
            }
        )
    return rows


def plot_scatter(df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    pairs = [
        ("log10_Re_proxy", "temporal_diff_entropy_mean"),
        ("log10_Ca_proxy", "temporal_diff_entropy_mean"),
        ("log10_Re_proxy", "entropy_delta_mean"),
        ("log10_Ca_proxy", "entropy_delta_mean"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (x_col, y_col) in zip(axes.ravel(), pairs):
        colors = np.where(df["correct_int"].to_numpy() == 1, "#2f6fbb", "#c73e3a")
        ax.scatter(df[x_col], df[y_col], c=colors, s=10, alpha=0.45, linewidths=0)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Spearman={corr(df, x_col, y_col, 'spearman'):.3f}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_re_ca_entropy_scatter.png", dpi=250)
    plt.close(fig)


def write_report(summary: dict, path: Path) -> None:
    def fmt(value: float | None) -> str:
        return "null" if value is None else f"{value:.4f}"

    lines = [
        "# Re/Ca Entropy-Correlation Probe",
        "",
        "## Definition",
        "",
        "- `mu = density * kinematic_viscosity` because raw dynamic-viscosity files contain unit inconsistencies.",
        "- `omega = 2*pi*RPM/60`.",
        "- `Re_proxy = omega / kinematic_viscosity`; true Re differs by a constant length-squared factor.",
        "- `Ca_proxy = mu * omega / surface_tension`; true Ca differs by a constant length factor.",
        "- `We_proxy = density * omega^2 / surface_tension`, equivalent to `Re_proxy * Ca_proxy` under the same omitted geometry constants.",
        "- Constant geometry factors do not change correlations.",
        "",
        "## Model Accuracy Reference",
        "",
        f"- Video rows: `{summary['video']['rows']}`; model accuracy: `{summary['video']['accuracy']}`",
        f"- Window rows: `{summary['window']['rows']}`; model accuracy: `{summary['window']['accuracy']}`",
        "",
        "## Strongest Re/Ca vs Entropy Correlations: Video Level",
        "",
    ]
    for row in summary["video"]["physics_entropy_correlations"][:12]:
        lines.append(
            f"- `{row['physics_feature']}` vs `{row['entropy_feature']}`: "
            f"Spearman `{fmt(row['spearman'])}`, Pearson `{fmt(row['pearson'])}`"
        )
    lines.extend(["", "## Correctness Correlations: Video Level", ""])
    for row in summary["video"]["correctness_correlations"][:12]:
        lines.append(
            f"- `{row['feature']}` vs correct: Pearson `{fmt(row['pearson_with_correct'])}`, "
            f"mean correct `{fmt(row['mean_correct'])}`, mean wrong `{fmt(row['mean_wrong'])}`"
        )
    lines.extend(["", "## Strongest Re/Ca vs Entropy Correlations: Window Level", ""])
    for row in summary["window"]["physics_entropy_correlations"][:12]:
        lines.append(
            f"- `{row['physics_feature']}` vs `{row['entropy_feature']}`: "
            f"Spearman `{fmt(row['spearman'])}`, Pearson `{fmt(row['pearson'])}`"
        )
    lines.extend(["", "## Correctness Correlations: Window Level", ""])
    for row in summary["window"]["correctness_correlations"][:12]:
        lines.append(
            f"- `{row['feature']}` vs correct: Pearson `{fmt(row['pearson_with_correct'])}`, "
            f"mean correct `{fmt(row['mean_correct'])}`, mean wrong `{fmt(row['mean_wrong'])}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this as a diagnostic correlation study only. Because Re and Ca are functions of RPM, viscosity, density, and surface tension, high correlation with entropy does not prove causal model usage. Correctness correlations are the direct check for whether these values explain model failures.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    entropy_dir = Path(args.entropy_dir)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_params = load_raw_params(dataset_root)
    video_df = attach_physics(pd.read_csv(entropy_dir / "video_entropy_features.csv"), raw_params)
    window_df = attach_physics(pd.read_csv(entropy_dir / "window_entropy_features.csv"), raw_params)

    video_out = output_dir / "video_entropy_re_ca_features.csv"
    window_out = output_dir / "window_entropy_re_ca_features.csv"
    video_df.to_csv(video_out, index=False)
    window_df.to_csv(window_out, index=False)

    plot_scatter(video_df, output_dir, "video_level")
    plot_scatter(window_df, output_dir, "window_level")

    summary = {
        "entropy_dir": str(entropy_dir),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "dimensionless_proxy_note": (
            "Re_proxy and Ca_proxy omit constant geometry length factors; this preserves correlations."
        ),
        "video": {
            "rows": int(len(video_df)),
            "accuracy": float(video_df["correct_int"].mean()),
            "physics_entropy_correlations": correlation_table(video_df),
            "correctness_correlations": correctness_table(video_df),
            "accuracy_by_true_class": grouped_accuracy(video_df, "true_viscosity_class"),
            "accuracy_by_rpm": grouped_accuracy(video_df, "rpm_raw"),
            "artifacts": {
                "feature_csv": str(video_out),
                "scatter": str(output_dir / "video_level_re_ca_entropy_scatter.png"),
            },
        },
        "window": {
            "rows": int(len(window_df)),
            "accuracy": float(window_df["correct_int"].mean()),
            "physics_entropy_correlations": correlation_table(window_df),
            "correctness_correlations": correctness_table(window_df),
            "accuracy_by_true_class": grouped_accuracy(window_df, "true_viscosity_class"),
            "accuracy_by_rpm": grouped_accuracy(window_df, "rpm_raw"),
            "artifacts": {
                "feature_csv": str(window_out),
                "scatter": str(output_dir / "window_level_re_ca_entropy_scatter.png"),
            },
        },
    }
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, report_path)
    print(report_path)
    print(summary_path)


if __name__ == "__main__":
    main()
