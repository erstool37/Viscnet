import os, gc, torch, numpy as np
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def save_attention(encoder_ddp, frames, rpm_idx, checkpoint, names, output, target, patch=16):
    base = os.path.basename(checkpoint)
    run_tag = os.path.splitext(base)[0]  
    model = encoder_ddp.module if hasattr(encoder_ddp, "module") else encoder_ddp
    save_dir = f"src/inference/attention_maps/volumes/{run_tag}"
    os.makedirs(save_dir, exist_ok=True)

    # small CPU-only ops
    out = output.detach().cpu().numpy()
    tgt = target.detach().cpu().numpy()
    pred_cls, true_cls = int(np.argmax(out)), (int(np.argmax(tgt)) if tgt.size > 1 else int(tgt))

    # capture only the final attention we see
    last_attn = [None]
    handles = []
    def _hook(_m, _i, o):
        y = o[-1] if isinstance(o, (tuple, list)) else o
        if torch.is_tensor(y) and y.ndim == 4 and y.shape[-1] == y.shape[-2]:
            last_attn[0] = y.detach()  # keep on device for now (no grad)

    for n, m in model.named_modules():
        if "attn" in n.lower() or "attention" in n.lower():
            handles.append(m.register_forward_hook(_hook))

    # forward (inference mode)
    model.eval()
    _ = encoder_ddp(frames, rpm_idx)

    # compute shape facts from input (no extra tensors)
    _, T, _, H, W = frames.shape
    Hp, Wp = H // patch, W // patch
    ppf = Hp * Wp  # patches per frame

    # process attention mostly on GPU, move minimal tensor to CPU
    attn = last_attn[0]                    # [B, heads, tokens, tokens]
    if last_attn[0] is None:
        print("[warn] no attention captured; skipping"); return None
    attn = attn[0].mean(0)                 # [tokens, tokens]   (mean over heads)
    tokens = attn.shape[-1]
    cls_to_tokens = attn[0, 1:]            # [tokens-1]
    M = cls_to_tokens.shape[0]
    G = (M // ppf)
    if G > 0:
        cls_to_tokens = cls_to_tokens[:G * ppf]
        V = cls_to_tokens.reshape(G, Hp, Wp).permute(1, 2, 0)  # [Hp, Wp, G]
        # V = (V - V.min()) / (V.max() - V.min() + 1e-8)
        V_cpu = V.to(torch.float32).cpu().numpy()              # compact dtype
    else:
        V_cpu = np.zeros((Hp, Wp, 0), dtype=np.float16)

    # save + log
    path = os.path.join(save_dir, f"{names}_class{true_cls}_pred{pred_cls}_cls_attn_vol.npy")
    np.save(path, V_cpu)
    print(f"[attn] saved {path}  (Hp={Hp}, Wp={Wp}, G={G})")

    M = cls_to_tokens.numel()
    expected = (H // patch) * (W // patch) * (T/2)
    if M < expected:
        print(f"[warn] tokens {M} < expected {expected}; check tokenizer/tubelet layout")
        return None
    G = M // (Hp * Wp)
    cls_to_tokens = cls_to_tokens[:G * (Hp * Wp)]

    # --- cleanup: remove hooks, drop tensors, free VRAM/RAM ---
    for h in handles: h.remove()
    del handles, last_attn, attn, cls_to_tokens
    del V_cpu
    # drop references to big inputs/outputs
    del output, target
    # frames/rpm_idx may be reused by caller; at least drop our ref
    frames = None; rpm_idx = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return path


def viz_attention(checkpoint):
    import os, re, glob, gc
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Any, Dict, List, Tuple, Union

    base = os.path.basename(checkpoint)
    run_tag = os.path.splitext(base)[0]
    cls_tol = 0
    npy_dir = f"src/inference/attention_maps/volumes/{run_tag}"
    out_dir = f"src/inference/attention_maps/panels/{run_tag}"
    os.makedirs(out_dir, exist_ok=True)

    # ---------- regex patterns ----------
    rev = re.compile(r'visc([0-9]*\.?[0-9]+)')
    rrp = re.compile(r'rpm([0-9]+)')
    rcl = re.compile(r'_class([0-9]+)')
    rpd = re.compile(r'_pred([0-9]+)')

    # ---------- helpers ----------
    def _parse_meta(path: str) -> Tuple[Union[float, None], Union[int, None], Union[int, None], Union[int, None]]:
        b = os.path.basename(path)
        mv, mr, mt, mp = rev.search(b), rrp.search(b), rcl.search(b), rpd.search(b)
        visc = float(mv.group(1)) if mv else None
        rpm  = int(mr.group(1))   if mr else None
        tru  = int(mt.group(1))   if mt else None
        pred = int(mp.group(1))   if mp else None
        return visc, rpm, tru, pred

    def _is_correct(r: Dict[str, Any]) -> bool:
        return (r["pred"] is not None) and (r["true"] is not None) and (abs(r["pred"] - r["true"]) <= cls_tol)

    # ---------- normalization logic ----------
    def _normalize_per_slice(V: np.ndarray) -> np.ndarray:
        """Normalize per temporal slice (for spatial map averaging)."""
        vmin = np.nanmin(V, axis=(0,1), keepdims=True)
        vmax = np.nanmax(V, axis=(0,1), keepdims=True)
        return (V - vmin) / (vmax - vmin + 1e-8)

    def _normalize_global(V: np.ndarray) -> np.ndarray:
        """Normalize globally (for temporal curves)."""
        vmin, vmax = np.nanmin(V), np.nanmax(V)
        return (V - vmin) / (vmax - vmin + 1e-8)

    # ---------- metric builders ----------
    def _xy_map(V: np.ndarray) -> np.ndarray:
        """Spatial attention: per-slice normalized first, then averaged over time."""
        Vn = _normalize_per_slice(V)
        Hmap = np.nanmean(Vn, axis=2)
        return (Hmap - np.nanmin(Hmap)) / (np.nanmax(Hmap) - np.nanmin(Hmap) + 1e-8)

    def _t_curve(V: np.ndarray) -> np.ndarray:
        """Temporal attention: globally normalized first, then averaged over space."""
        Vn = _normalize_global(V)
        T = np.nanmean(Vn, axis=(0,1))
        return (T - np.nanmin(T)) / (np.nanmax(T) - np.nanmin(T) + 1e-8)

    def _nanpad_stack(vols: List[np.ndarray]) -> np.ndarray:
        Hm = max(v.shape[0] for v in vols)
        Wm = max(v.shape[1] for v in vols)
        Gm = max(v.shape[2] for v in vols)
        arr = np.full((len(vols), Hm, Wm, Gm), np.nan, dtype=float)
        for i, v in enumerate(vols):
            arr[i, :v.shape[0], :v.shape[1], :v.shape[2]] = v
        return arr

    def _mean_by_group(recs: List[Dict[str, Any]], key: str):
        """Aggregate volumes by grouping key."""
        groups: Dict[Any, List[np.ndarray]] = {}
        for r in recs:
            k = r[key]
            if k is not None:
                groups.setdefault(k, []).append(r["V"])

        maps, curves = {}, {}
        for k, lst in groups.items():
            arr = _nanpad_stack(lst)
            Vm = np.nanmean(arr, axis=0)
            maps[k] = _xy_map(Vm)
            curves[k] = _t_curve(Vm)
            del arr, Vm
            gc.collect()
        return maps, curves

    # ---------- plotting helpers ----------
    def _make_strip_plot(maps_c, curves_c, maps_w, curves_w, title_prefix, keys_sorted, xlabels):
        """Draw 3-row panel (correct spatial, wrong spatial, temporal curves combined)."""
        cols = max(1, len(keys_sorted))
        fig = plt.figure(figsize=(max(6, 2.6*cols), 6.4))
        gs = fig.add_gridspec(3, cols, height_ratios=[1,1,1.4])

        # Row 0: correct maps
        for i, k in enumerate(keys_sorted):
            ax = fig.add_subplot(gs[0, i])
            if k in maps_c:
                ax.imshow(maps_c[k], cmap="viridis", origin="upper", vmin=0, vmax=1)
            ax.set_title(f"{title_prefix}={xlabels[i]}\nCORRECT", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

        # Row 1: wrong maps
        for i, k in enumerate(keys_sorted):
            ax = fig.add_subplot(gs[1, i])
            if k in maps_w:
                ax.imshow(maps_w[k], cmap="viridis", origin="upper", vmin=0, vmax=1)
            ax.set_title(f"{title_prefix}={xlabels[i]}\nWRONG", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

        # Row 2: temporal curves (combined C/W)
        ax_t = fig.add_subplot(gs[2, :])
        for k in keys_sorted:
            if k in curves_c:
                ax_t.plot(curves_c[k], lw=1.4, marker="o", ms=3, label=f"{k} (C)")
            if k in curves_w:
                ax_t.plot(curves_w[k], lw=1.4, ls="--", marker="x", ms=3, label=f"{k} (W)")
        ax_t.set_xlabel("Tubelet Time Index (g)")
        ax_t.set_ylabel("Mean Attention (Globally Norm.)")
        ax_t.grid(True, alpha=0.3, ls="--")
        if keys_sorted:
            ax_t.legend(fontsize=7, ncol=min(6, len(keys_sorted)), loc='upper center', bbox_to_anchor=(0.5,-0.18))
        fig.suptitle(f"{title_prefix}-wise Attention Analysis (Slice-norm Spatial / Global-norm Temporal)", y=0.99)
        plt.tight_layout(rect=[0,0,1,0.93])
        out_png = os.path.join(out_dir, f"{title_prefix}_temporalSpatial_norm_strips.png")
        plt.savefig(out_png, dpi=220)
        plt.close(fig)
        print(f"[OK] Saved {title_prefix} → {out_png}")

    def _plot_temporal_only(curves: Dict[Any, np.ndarray], title_prefix: str, keys_sorted: List[Any], xlabels: List[str], subset_tag: str):
        """Temporal-only plot for either correct or wrong groups."""
        if not keys_sorted:
            return
        fig = plt.figure(figsize=(max(6, 2.6), 3.6))
        ax = fig.add_subplot(1,1,1)
        for k in keys_sorted:
            if k in curves:
                ax.plot(curves[k], lw=1.6, marker="o", ms=3, label=f"{k}")
        ax.set_xlabel("Tubelet Time Index (g)")
        ax.set_ylabel("Mean Attention (Globally Norm.)")
        ax.set_title(f"{title_prefix}: Temporal ({subset_tag.upper()})")
        ax.grid(True, alpha=0.3, ls="--")
        ax.legend(fontsize=7, ncol=min(6, len(keys_sorted)))
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{title_prefix}_temporal_{subset_tag.lower()}.png")
        plt.savefig(out_png, dpi=220)
        plt.close(fig)
        print(f"[OK] Saved {title_prefix} temporal ({subset_tag}) → {out_png}")

    # ---------- load all npy ----------
    paths = glob.glob(os.path.join(npy_dir, "**", "*_cls_attn_vol.npy"), recursive=True)
    print(f"[INFO] Found {len(paths)} NPY files in '{npy_dir}'")
    records: List[Dict[str, Any]] = []

    for p in sorted(paths):
        try:
            V = np.load(p)
            if V.ndim != 3: 
                continue
            visc, rpm, tru, pred = _parse_meta(p)
            if tru is None or pred is None: 
                continue
            records.append(dict(V=V.astype(np.float32), visc=visc, rpm=rpm, true=int(tru), pred=int(pred)))
        except Exception:
            continue

    if not records:
        print("[FATAL] No usable attention volumes."); 
        return

    # ---------- split ----------
    recs_c = [r for r in records if _is_correct(r)]
    recs_w = [r for r in records if not _is_correct(r)]
    print(f"[INFO] Correct: {len(recs_c)} | Wrong: {len(recs_w)}")

    # ---------- viscosity ----------
    maps_c, curves_c = _mean_by_group(recs_c, "visc")
    maps_w, curves_w = _mean_by_group(recs_w, "visc")
    keys_v = sorted(set(list(maps_c.keys()) + list(maps_w.keys())))
    _make_strip_plot(maps_c, curves_c, maps_w, curves_w, "Viscosity", keys_v, [f"{k:.4f}" for k in keys_v])
    _plot_temporal_only(curves_c, "Viscosity", keys_v, [f"{k:.4f}" for k in keys_v], subset_tag="correct")
    _plot_temporal_only(curves_w, "Viscosity", keys_v, [f"{k:.4f}" for k in keys_v], subset_tag="wrong")

    # ---------- rpm ----------
    maps_c, curves_c = _mean_by_group(recs_c, "rpm")
    maps_w, curves_w = _mean_by_group(recs_w, "rpm")
    keys_r = sorted(set(list(maps_c.keys()) + list(maps_w.keys())))
    _make_strip_plot(maps_c, curves_c, maps_w, curves_w, "RPM", keys_r, [f"{int(k)}" for k in keys_r])
    _plot_temporal_only(curves_c, "RPM", keys_r, [f"{int(k)}" for k in keys_r], subset_tag="correct")
    _plot_temporal_only(curves_w, "RPM", keys_r, [f"{int(k)}" for k in keys_r], subset_tag="wrong")

    gc.collect()
    print("[DONE] Spatial (slice-norm) & Temporal (global-norm) panels complete, with separate temporal plots for correct/wrong.")