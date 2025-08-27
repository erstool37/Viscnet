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

def confusion_matrix(name, y_pred_or_logits, y_true, normalize=True, class_names=None, save_dir="src/inference/confusion_matrix"):
    y_true = np.asarray(y_true, dtype=int).ravel()
    X = np.asarray(y_pred_or_logits)

    # logits -> labels, labels -> as-is
    if X.ndim == 2:
        y_pred = X.argmax(axis=1).astype(int).ravel()
        labels = np.arange(X.shape[1])  # show all classes 0..C-1
    else:
        y_pred = X.astype(int).ravel()
        # show only observed classes unless class_names is provided
        labels = np.unique(np.concatenate([y_true, y_pred])) if class_names is None else np.arange(len(class_names))

    cm = sk_cm(y_true, y_pred, labels=labels, normalize=('true' if normalize else None))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(class_names if class_names is not None else labels))

    os.makedirs(save_dir, exist_ok=True)
    disp.plot(cmap="Blues", xticks_rotation=45, colorbar=True)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

import os, importlib, numpy as np, matplotlib.pyplot as plt
import torch

def plot_error_distribution(name, preds_all, tgts_all, descaler, path, save_dir="src/inference/error_plots"):
    """
    Draw error distribution for dynamic viscosity only.
    x-axis: true dynamic viscosity (unnormalized)
    y-axis: % error = 100 * (pred - true) / true
    """
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

    # unnormalize dynamic viscosity only (column 1)
    P_dyn = f(P_t[:, 1], "dynamic_viscosity", path).unsqueeze(-1)
    T_dyn = f(T_t[:, 1], "dynamic_viscosity", path).unsqueeze(-1)

    # percentage error
    eps = 1e-12
    err_dyn = 100.0 * (P_dyn - T_dyn) / (T_dyn + eps)

    # convert to numpy
    true_vals = T_dyn.detach().cpu().numpy().ravel()
    err_vals  = err_dyn.detach().cpu().numpy().ravel()

    # filter out errors with absolute value > 100
    mask = np.abs(err_vals) <= 100
    true_vals = true_vals[mask]
    err_vals  = err_vals[mask]

    # scatter plot
    plt.figure(figsize=(6, 5))
    plt.scatter(true_vals, err_vals, s=10, alpha=0.6, color="tab:blue")
    plt.axhline(0.0, lw=1, color="k")
    plt.xlabel("True Dynamic Viscosity")
    plt.ylabel("Error (%)")
    plt.title("Error Distribution: Dynamic Viscosity")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")