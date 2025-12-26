import torch
import torch.nn as nn
import wandb
import os.path as osp
import json
import importlib
from torch.nn import functional as F

class MAPE(nn.Module):
    """
    unnormalized and MAPE calculation
    """
    def __init__(self, unnormalizer, path, smooth_label):
        super(MAPE, self).__init__()
        self.unnormalizer = unnormalizer
        self.path = path

    def forward(self, pred, target):
        utils = importlib.import_module("utils")
        descaler = getattr(utils, self.unnormalizer)

        pred_kinvisc = descaler(pred[:, 2], "kinematic_viscosity", self.path).unsqueeze(-1).to(pred.device)
        target_kinvisc = descaler(target[:, 2], "kinematic_viscosity", self.path).unsqueeze(-1).to(pred.device)
        # loss_dynvisc = F.mse_loss(pred_dynvisc, target_dynvisc)/target_dynvisc.unsqueeze(-1)
        loss_kinvisc = (torch.abs(pred_kinvisc - target_kinvisc) / target_kinvisc).unsqueeze(-1)

        total_loss = loss_kinvisc

        return total_loss
    

        # pred_den = descaler(pred[:,0], "density", self.path).unsqueeze(-1).to(pred.device)
        # pred_surfT = descaler(pred[:,2], "surface_tension", self.path).unsqueeze(-1).to(pred.device)
        # target_den = descaler(target[:,0], "density", self.path).unsqueeze(-1).to(pred.device)
        # target_surfT = descaler(target[:,2], "surface_tension", self.path).unsqueeze(-1).to(pred.device)
        # loss_den = torch.mean((torch.abs(pred_den - target_den) / target_den)).unsqueeze(-1)
        # loss_surfT = torch.mean((torch.abs(pred_surfT - target_surfT) / target_surfT)).unsqueeze(-1)