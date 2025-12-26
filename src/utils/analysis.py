import torch
import torch.nn.functional as F
import numpy as np
import os
import seaborn as sns
import importlib
import wandb
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_cm, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import csv

def distribution(data, ref=None, title='Normalized Value Distribution', save_path='.', prefix='dist'):
    import os
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    plt.figure()
    ax = sns.histplot(data, kde=True, bins=50, stat='density', edgecolor='black')
    if ref is not None:
        ymax = ax.get_ylim()[1]
        plt.vlines(ref, ymin=0, ymax=ymax, color='red', linestyle='--', label=f'Ref: {ref}')
        plt.legend()
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(save_path)
    plt.close()

def MAPEtestcalculator(pred, target, descaler, method, path):
    utils = importlib.import_module("utils")
    descaler = getattr(utils, descaler)
    
    pred_den = descaler(pred[:,0], "density", path).unsqueeze(-1)
    pred_dynvisc = descaler(pred[:,1], "dynamic_viscosity", path).unsqueeze(-1)
    pred_surfT = descaler(pred[:,2], "surface_tension", path).unsqueeze(-1)

    target_den = descaler(target[:,0], "density", path).unsqueeze(-1)
    target_dynvisc = descaler(target[:,1], "dynamic_viscosity", path).unsqueeze(-1)
    target_surfT = descaler(target[:,2], "surface_tension", path).unsqueeze(-1)

    loss_mape_den = (torch.abs(pred_den - target_den) / target_den) * 100
    loss_mape_dynvisc = (torch.abs(pred_dynvisc - target_dynvisc) / target_dynvisc) * 100
    loss_mape_surfT = (torch.abs(pred_surfT - target_surfT) / target_surfT) * 100
    
    error = torch.cat([loss_mape_den, loss_mape_dynvisc, loss_mape_surfT], dim=1) 

    wandb.log({"predict visc": pred_dynvisc, "answer visc": target_dynvisc})
    return error

def confusion_matrix(name, y_pred_or_logits, y_true, normalize=True, class_names=None, save_dir="src/inference/confusion_matrix", vmax=1.0):
    print("Start drawing confusion matrix")
    y_true = np.asarray(y_true, dtype=int).ravel()
    X = np.asarray(y_pred_or_logits)

    if X.ndim == 2:
        y_pred = X.argmax(axis=1).astype(int).ravel()
        labels = np.arange(X.shape[1])
    else:
        y_pred = X.astype(int).ravel()
        labels = np.unique(np.concatenate([y_true, y_pred])) if class_names is None else np.arange(len(class_names))

    # 🔹 Compute accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"Accuracy: {accuracy:.4f}")

    cm = sk_cm(y_true, y_pred, labels=labels, normalize=('true' if normalize else None))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(class_names if class_names is not None else labels))

    os.makedirs(save_dir, exist_ok=True)
    disp.plot(cmap="Blues", xticks_rotation=45, colorbar=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=(class_names if class_names is not None else labels))

    disp.plot(cmap="Blues", xticks_rotation=45, colorbar=True)
    disp.im_.set_clim(0, vmax)

    # write numbers in matrix as .2f
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         disp.ax_.text(j, i, f"{cm[i, j]:.2f}",
    #                     ha="center", va="center",
                        # color="black", fontsize=8)

    disp.im_.set_clim(0, vmax)

    # Show accuracy in title
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=1000)
    plt.close()

def plot_error_distribution(
    name, preds_all, tgts_all, descaler, path,
    save_dir="src/inference/error_plots", topk=10
):
    os.makedirs(save_dir, exist_ok=True)

    # -------- 1) Fix pairs: flatten & enforce 1:1 alignment --------
    preds_list = [np.atleast_2d(p).astype(np.float32) for p in preds_all]
    tgts_list  = [np.atleast_2d(t).astype(np.float32) for t in tgts_all]
    if not preds_list or not tgts_list:
        print("No data to plot.")
        return

    P = np.vstack(preds_list)
    T = np.vstack(tgts_list)
    n = min(len(P), len(T))
    P, T = P[:n], T[:n]   # enforce alignment

    # -------- 2) Torch tensors --------
    P_t = torch.from_numpy(P)
    T_t = torch.from_numpy(T)

    # -------- 3) Descale kinematic viscosity column (index 2) --------
    utils = importlib.import_module("utils")
    f = getattr(utils, descaler)

    P_kin = f(P_t[:, 2], "kinematic_viscosity", path).unsqueeze(-1)  # [N,1]
    T_kin = f(T_t[:, 2], "kinematic_viscosity", path).unsqueeze(-1)  # [N,1]

    # -------- 4) Sort by target, keep pairs synced --------
    sort_idx = torch.argsort(T_kin[:, 0])      # ascending
    T_kin = T_kin[sort_idx]
    P_kin = P_kin[sort_idx]

    # -------- 5) Errors --------
    eps = 1e-12
    err_abs = torch.abs(P_kin - T_kin)                       # absolute error
    err_pct = 100.0 * err_abs / (T_kin.clamp_min(eps))       # percentage error

    # -> numpy
    t_np = T_kin.squeeze(1).detach().cpu().numpy()
    p_np = P_kin.squeeze(1).detach().cpu().numpy()
    e_np = err_pct.squeeze(1).detach().cpu().numpy()

    # clean mask
    mask = np.isfinite(t_np) & np.isfinite(p_np) & np.isfinite(e_np) & (np.abs(e_np) <= 1000)
    t_np, p_np, e_np = t_np[mask], p_np[mask], e_np[mask]

    # -------- 6) Summary / answer info --------
    mae = np.mean(np.abs(p_np - t_np))
    mape = np.mean(np.abs((p_np - t_np) / np.clip(t_np, eps, None))) * 100.0
    rmse = np.sqrt(np.mean((p_np - t_np) ** 2))
    npts = len(t_np)

    # top-k worst (by % error)
    topk = min(topk, npts)
    worst_idx = np.argsort(-np.abs(e_np))[:topk]
    worst_tbl = pd.DataFrame({
        "target": t_np[worst_idx],
        "pred":   p_np[worst_idx],
        "err_%":  e_np[worst_idx],
    })

    # save full table (sorted by target) for traceability
    df_all = pd.DataFrame({"target": t_np, "pred": p_np, "err_%": e_np})
    csv_path = os.path.join(save_dir, f"{name}_sorted_target.csv")
    df_all.to_csv(csv_path, index=False)

    # -------- 7) Plot diagram (error vs sorted true) + info box --------
    plt.figure(figsize=(7, 5.2))
    plt.scatter(t_np, e_np, s=10, alpha=0.6)
    plt.axhline(0.0, lw=1, color="k")
    plt.xlabel("True Kinematic Viscosity (sorted)")
    plt.ylabel("Error (%)")
    plt.title("Error Distribution: Kinematic Viscosity")

    # info box on plot
    infostr = (
        f"N={npts}\n"
        f"MAE={mae:.4g}\n"
        f"RMSE={rmse:.4g}\n"
        f"MAPE={mape:.2f}%\n"
        f"Saved CSV: {os.path.basename(csv_path)}"
    )
    plt.gcf().text(0.98, 0.52, infostr, ha="right", va="top",
                   fontsize=9, bbox=dict(boxstyle="round", fc="w", ec="0.6", alpha=0.9))

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Print
    print(f"[{name}] N={npts}  MAE={mae:.4g}  RMSE={rmse:.4g}  MAPE={mape:.2f}%")
    print(f"Saved plot: {out_path}")
    print(f"Saved table: {csv_path}")
    if topk > 0:
        print(f"Top-{topk} worst by % error:")
        print(worst_tbl.to_string(index=False))


def csv_export(logits, energy, csv_path="results.csv"):
    L = torch.as_tensor(logits, dtype=torch.float32).detach().cpu()
    L = L.view(-1, L.shape[-1])                 # [N,C] even if inputs were [C]
    E = torch.as_tensor(energy, dtype=torch.float32).detach().cpu().view(-1)  # [N]

    P = F.softmax(L, dim=1)                     # probs
    pred = P.argmax(1)                          # [N]
    top_vals, top_idx = torch.topk(P, k=3, dim=1)  # [N,3] each

    df = pd.DataFrame({
        "pred_class": pred.numpy(),
        "energy":     E.numpy(),
        "top1_class": top_idx[:, 0].numpy(),
        "top1_prob":  top_vals[:, 0].numpy(),
        "top2_class": top_idx[:, 1].numpy(),
        "top2_prob":  top_vals[:, 1].numpy(),
        "top3_class": top_idx[:, 2].numpy(),
        "top3_prob":  top_vals[:, 2].numpy(),
    })
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

def reliability_diagram(preds_local, tgts_local, name="test"):
    """Draw and save a reliability diagram (ECE + MCE) for classification."""
    n_bins = 15
    title = "Test Reliability Diagram"

    # Convert lists to tensors
    if isinstance(preds_local[0], torch.Tensor):
        logits = torch.cat([p.detach().cpu().reshape(1, -1) for p in preds_local], dim=0)
        onehot = torch.cat([t.detach().cpu().reshape(1, -1) for t in tgts_local], dim=0)
    else:
        logits = torch.tensor(np.vstack(preds_local))
        onehot = torch.tensor(np.vstack(tgts_local))

    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    true = onehot.argmax(dim=1)
    correct = (pred == true).float()

    edges = torch.linspace(0, 1, steps=n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_acc, bin_conf, ece, mce = [], [], 0.0, 0.0
    N = logits.size(0)

    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        mask = (conf > lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bin_acc.append(0.0)
            bin_conf.append((lo + hi) / 2)
            continue
        acc_i = float(correct[mask].mean())
        conf_i = float(conf[mask].mean())
        gap = abs(acc_i - conf_i)
        ece += (cnt / N) * gap
        mce = max(mce, gap)
        bin_acc.append(acc_i)
        bin_conf.append(conf_i)

    # Plot and save
    plt.figure(figsize=(5, 5), dpi=1000)
    plt.bar(centers.numpy(), bin_acc, width=1 / n_bins * 0.9, alpha=0.6,
            edgecolor="black", linewidth=0.5, label="Bin accuracy")
    plt.plot(centers.numpy(), bin_conf, marker="o", linewidth=1.5, label="Avg confidence")
    plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfect calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"{title}\nECE={ece:.3f}, MCE={mce:.3f}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"src/inference/reliability_plots/{name}.png", dpi=1000, bbox_inches="tight")
    plt.close()
    print(f"Saved reliability diagram → reliability_diagram.png (ECE={ece:.3f}, MCE={mce:.3f})")
