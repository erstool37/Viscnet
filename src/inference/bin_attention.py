import os
import re
import glob
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 8


NPY_DIR = "src/inference/attentiongraph/volumes/35000_finetune993_smallerVivit_layer10_1026_v0"
OUT_DIR = "src/inference/attentiongraph/panels"
CSV_PATH = "src/dataanalysis1027.csv"
VALID_TXT = "src/inference/reserved_1000_list.txt"

METRIC_COL = "rpm"
NUM_BINS = 5
BIN_MODE = "quantile"

os.makedirs(OUT_DIR, exist_ok=True)


rev = re.compile(r"visc(\d+\.\d+)")
rrp = re.compile(r"rpm(\d+)")
rcl = re.compile(r"(?:^|[_-])class(\d+)(?=[_.-]|$)")
rpd = re.compile(r"(?:^|[_-])pred(\d+)(?=[_.-]|$)")


def parse_meta(path: str):
    b = os.path.basename(path)
    mv, mr, mt, mp = rev.search(b), rrp.search(b), rcl.search(b), rpd.search(b)
    visc = float(mv.group(1)) if mv else None
    rpm = int(mr.group(1)) if mr else None
    tru = int(mt.group(1)) if mt else None
    pred = int(mp.group(1)) if mp else None
    return visc, rpm, tru, pred


with open(VALID_TXT, "r") as f:
    valid_names = {line.strip() for line in f if line.strip()}

names_list = list(valid_names)
counter = Counter(names_list)
dupes = {n: c for n, c in counter.items() if c > 1}


paths = glob.glob(os.path.join(NPY_DIR, "**", "*_cls_attn_vol.npy"), recursive=True)

records = []
for p in sorted(paths):
    try:
        V = np.load(p)
        if V.ndim != 3:
            continue
        vmin, vmax = V.min(), V.max()
        if vmax > vmin:
            V = (V - vmin) / (vmax - vmin)
        else:
            V = np.zeros_like(V, dtype=float)
    except Exception:
        continue

    visc, rpm, tru, pred = parse_meta(p)
    records.append(dict(V=V, visc=visc, rpm=rpm, true=tru, pred=pred, path=p))

if not records:
    raise RuntimeError(f"No valid volumes found under {NPY_DIR}")


df_metric = pd.read_csv(CSV_PATH)

if "name" not in df_metric.columns or METRIC_COL not in df_metric.columns:
    raise ValueError(f"{CSV_PATH} must contain 'name' and '{METRIC_COL}' columns.")

name_to_metric = {
    str(n): float(v)
    for n, v in zip(df_metric["name"], df_metric[METRIC_COL])
}


def match_name_to_metric(path: str):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if stem in name_to_metric:
        return name_to_metric[stem]
    stem2 = stem.replace("_cls_attn_vol", "")
    if stem2 in name_to_metric:
        return name_to_metric[stem2]
    for n, v in name_to_metric.items():
        if n in base:
            return v
    return None


kept = []
for r in records:
    mv = match_name_to_metric(r["path"])
    if mv is None or np.isnan(mv):
        continue
    r["metric"] = float(mv)
    kept.append(r)

records = kept

if not records:
    raise RuntimeError(f"No records with {METRIC_COL} matched.")


def t_curve(V):
    T = np.nanmean(V, axis=(0, 1))
    tmin, tmax = np.nanmin(T), np.nanmax(T)
    return (T - tmin) / (tmax - tmin + 1e-8)


def mean_temporal_by_group(recs, key):
    groups = {}
    for r in recs:
        if r.get(key) is None:
            continue
        groups.setdefault(r[key], []).append(r)

    curves = {}
    for k, lst in groups.items():
        Vs = [np.asarray(d["V"], dtype=float) for d in lst]
        if not Vs:
            continue

        Vs = [
            v if v.ndim == 3 else np.reshape(v, (1,) * (3 - v.ndim) + v.shape)
            for v in Vs
        ]

        Hm = max(v.shape[0] for v in Vs)
        Wm = max(v.shape[1] for v in Vs)
        Gm = max(v.shape[2] for v in Vs)

        padded = []
        for v in Vs:
            H, W, G = v.shape
            out = np.full((Hm, Wm, Gm), np.nan)
            out[:H, :W, :G] = v
            padded.append(out)

        arr = np.stack(padded, axis=0)
        Vm = np.nanmean(arr, axis=0)
        curves[k] = t_curve(Vm)

    return curves


def plot_temporal_by_rpm(curves, keys_sorted):
    if not curves:
        return

    G = len(next(iter(curves.values())))
    xs = np.arange(G)

    fig = plt.figure(figsize=(100 / 25.4, 70 / 25.4))
    ax = fig.add_subplot(1, 1, 1)

    for rpm in keys_sorted:
        ax.plot(xs, curves[rpm], lw=0.8, label=f"RPM={rpm:g}")

    ax.set_xlabel("Tubelet Time Index")
    ax.set_ylabel("Mean Attention")
    ax.grid(False)

    leg = ax.legend(fontsize=7, loc="upper right", frameon=True)
    leg.get_frame().set_linewidth(0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "rpm_temporal.png")
    plt.savefig(out_png, dpi=1000)
    plt.close(fig)


def _fmt_edge(x):
    if x == 0:
        return "0"
    s = f"{x:.3e}"
    a, b = s.split("e")
    a = a.rstrip("0").rstrip(".")
    b = int(b)
    return f"{a}×10^{{{b}}}"


vals = np.array([r["metric"] for r in records], dtype=float)

if BIN_MODE.lower() == "quantile":
    try:
        bin_idx, edges = pd.qcut(
            vals, q=NUM_BINS, labels=False, retbins=True, duplicates="drop"
        )
    except Exception:
        bin_idx = pd.cut(vals, bins=NUM_BINS, labels=False, include_lowest=True)
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        edges = np.linspace(vmin, vmax, NUM_BINS + 1)
else:
    bin_idx = pd.cut(vals, bins=NUM_BINS, labels=False, include_lowest=True)
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    edges = np.linspace(vmin, vmax, NUM_BINS + 1)

for r, bi in zip(records, bin_idx):
    r["mbin"] = None if bi is None or (isinstance(bi, float) and np.isnan(bi)) else int(bi)

keys_sorted = sorted({r["rpm"] for r in records if r["rpm"] is not None})
curves = mean_temporal_by_group(records, "rpm")
plot_temporal_by_rpm(curves, keys_sorted)
