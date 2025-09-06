import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import importlib
import wandb

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

def visualize_logits(logits_batch):
    print((len(logits_batch)))
    cmap='viridis'
    title='Batch Logits Heatmap'
    figsize=(12, 6)
    v = np.array(logits_batch).squeeze()
    
    # Find index of max logit in each row
    max_indices = np.argmax(v, axis=1)  # shape (N,)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(v, aspect='auto', cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Logit Value')
    
    # Overlay red squares on max positions
    for i, j in enumerate(max_indices):
        ax.scatter(j, i, s=50, facecolors='none', edgecolors='red', linewidths=1.5)
    
    ax.set_xlabel('Class Index')
    ax.set_ylabel('Sample Index')
    ax.set_title(title)
    fig.savefig("src/inference/heatmap.png", dpi=300)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_cm, ConfusionMatrixDisplay

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

    disp.im_.set_clim(0, vmax)

    # 🔹 Show accuracy in title
    plt.title(f"Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

import os, importlib, numpy as np, matplotlib.pyplot as plt
import torch

def plot_error_distribution(name, preds_all, tgts_all, descaler, path, save_dir="src/inference/error_plots"):
    os.makedirs(save_dir, exist_ok=True)

    # flatten gathered lists -> (N, D) arrays
    P = np.vstack([np.atleast_2d(p) for p in preds_all]).astype(np.float32)
    T = np.vstack([np.atleast_2d(t) for t in tgts_all]).astype(np.float32)
    print(P.shape, T.shape)
    # convert to torch
    P_t = torch.from_numpy(P)
    T_t = torch.from_numpy(T)
    print(P_t.shape, T_t.shape)

    # load descaler from utils
    utils = importlib.import_module("utils")
    f = getattr(utils, descaler)

    # unnormalize kinematic viscosity only (column 1)
    P_kin = f(P_t[:, 2], "kinematic_viscosity", path).unsqueeze(-1)
    T_kin = f(T_t[:, 2], "kinematic_viscosity", path).unsqueeze(-1)

    # percentage error
    err_kin = 100.0 * abs(P_kin - T_kin) / (T_kin)

    # convert to numpy
    true_vals = T_kin.detach().cpu().numpy().ravel()
    err_vals  = err_kin.detach().cpu().numpy().ravel()

    print(true_vals)

    # filter out errors with absolute value > 100
    mask = np.abs(err_vals) <= 1000
    true_vals = true_vals[mask]
    err_vals  = err_vals[mask]

    # scatter plot
    plt.figure(figsize=(6, 5))
    plt.scatter(true_vals, err_vals, s=10, alpha=0.6, color="tab:blue")
    plt.axhline(0.0, lw=1, color="k")
    plt.xlabel("True Kinematic Viscosity")
    plt.ylabel("Error (%)")
    plt.title("Error Distribution: Kinematic Viscosity")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def new_plot_error_distribution(name, preds_all, tgts_all, descaler, path, save_dir="src/inference/error_plots"):
    os.makedirs(save_dir, exist_ok=True)

    # flatten gathered lists -> (N, D) arrays
    P = np.vstack([np.atleast_2d(p) for p in preds_all]).astype(np.float32)
    T = np.vstack([np.atleast_2d(t) for t in tgts_all]).astype(np.float32)

    # convert to torch
    P_t = torch.from_numpy(P)
    T_t = torch.from_numpy(T)

    # load descaler from utils
    utils = importlib.import_module("utils")
    f = getattr(utils, descaler)

    # unnormalize viscosities
    P_kin = f(P_t[:, 2], "kinematic_viscosity", path).unsqueeze(-1)  # predictions
    T_kin = f(T_t[:, 2], "kinematic_viscosity", path).unsqueeze(-1)  # true values

    # convert to numpy
    pred_vals = P_kin.detach().cpu().numpy().ravel()
    true_vals = T_kin.detach().cpu().numpy().ravel()

    # reference viscosity values
    ref_viscs = [
        8.955241763971034e-07,
        1.3606565275983135e-05,
        1.800701766707428e-05,
        2.3848869381823438e-05,
        3.160701316546058e-05,
        4.191336494974671e-05,
        5.560892086883875e-05,
        7.381311130517014e-05,
        9.80162226060615e-05,
        0.000130202569702055,
    ]

    # black reference lines (test values)
    black_refs = {
        8.955241763971034e-07,
        1.800701766707428e-05,
        3.160701316546058e-05,
        5.560892086883875e-05,
        9.80162226060615e-05,
    }

    # plot predictions
    x = np.arange(len(pred_vals))
    plt.figure(figsize=(10, 6))
    plt.plot(x, pred_vals, "o", markersize=3, alpha=0.6, color="tab:blue", label="Predicted Viscosity")

    # add horizontal + vertical lines whenever true value changes
    last_val = true_vals[0]
    for i in range(1, len(true_vals)):
        if true_vals[i] != last_val:
            plt.axvline(i, color="red", linestyle="--", alpha=0.7)   # vertical red line
            last_val = true_vals[i]

    # add reference viscosity lines with dummy handles for legend
    for v in ref_viscs:
        if v in black_refs:
            plt.axhline(v, color="black", linestyle="-", alpha=0.7, label="Test Provided Values")
        else:
            plt.axhline(v, color="blue", linestyle="-", alpha=0.5, label="Train Provided Values")

    # add legend handles for train/test references
    plt.axhline(0.00015, color="black", linestyle="-", alpha=0.7, label="Test Provided Values")
    plt.axhline(0.00015, color="blue", linestyle="-", alpha=0.5, label="Train Provided Values")

    # labels
    plt.ylabel("Kinematic Viscosity (SI units)")
    plt.xlabel("Sample Index")
    plt.title("Predicted Viscosity with Ground Truth and Reference Lines")
    plt.legend()
    plt.tight_layout()

    # save
    out_path = os.path.join(save_dir, f"{name}_visc.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")