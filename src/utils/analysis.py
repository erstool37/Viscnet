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