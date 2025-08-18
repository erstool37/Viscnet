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

def confusion_matrix(name, logits_batch, true_labels, normalize=True):
    y_pred = np.argmax(logits_batch, axis=1)      # shape (N,)
    y_true = true_labels.astype(int).flatten()     # already integer labels

    cm = sk_cm(y_true, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"src/inference/confusion_matrix/{name}.png", dpi=300)
    plt.close()