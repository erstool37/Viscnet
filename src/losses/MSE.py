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

        loss_den = loss[:, 0].mean()
        loss_dynvisc = loss[:, 1].mean()
        loss_surfT = loss[:, 2].mean()

        loss_total = loss_dynvisc

        # wandb.log({
        #     "loss_den": loss_den,
        #     "loss_visc": loss_dynvisc,
        #     "loss_surf": loss_surfT
        # })

        return loss_total