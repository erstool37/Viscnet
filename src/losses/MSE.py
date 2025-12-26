import torch
import torch.nn as nn
import wandb

class MSE(nn.Module):
    """
    Simple MSE loss for normalized data
    """
    def __init__(self, unnormalizer=None, path=None):
        super(MSE, self).__init__()

    def forward(self, pred, target):
        loss = (pred[:,:3] - target[:, :3]) ** 2
        loss_kinvisc = loss[:, 1].mean()
        loss_total = loss_kinvisc

        return loss_total