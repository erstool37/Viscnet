import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
import importlib

class MSLE(nn.Module):
    """
    simple MSLE loss for normalized data
    """
    def __init__(self, unnormalizer, path):
        super(MSLE, self).__init__()

    def forward(self, pred, target):        
        loss = (torch.log1p(pred[:,:3]) - torch.log1p(target[:,:3])) ** 2 
        
        loss_kinvisc = loss[:, 1].mean()

        loss_total = loss_kinvisc

        return loss_total










    # Intricate loss function, linking density and dynviscosity using real world kknematic viscosity, result : fail. unstable training
    """
        def forward(self, pred, target):
        pred = torch.clamp(pred, min=0)
        target = torch.clamp(target, min=0)

        # predicted kinVisc calculation in unnormalized version
        pred_den = logdescaler(pred[:,0], "density").unsqueeze(-1).to(pred.device)
        pred_dynVisc = logdescaler(pred[:,1], "dynamic_viscosity").unsqueeze(-1).to(pred.device)
        
        pred_kinVisc = (pred_dynVisc / pred_den)

        pred = torch.cat((pred, pred_kinVisc), dim=-1)

        # real kinVisc calculation
        ans_kinVisc = logdescaler(pred[:,3], "kinematic_viscosity")
        target[:,3] = ans_kinVisc

        loss = (torch.log1p(pred) - torch.log1p(target)) ** 2 
        # loss = (pred-target) ** 2

        loss_den = loss[:, 0].mean()
        loss_dynvisc = loss[:, 1].mean()
        loss_surfT = loss[:, 2].mean()
        loss_kinvisc = loss[:, 3].mean()
        # loss_kinvisc = F.mse_loss(pred[:,3], target[:,3]) # already mean

        loss_total = 5 * loss_den +  1 * loss_surfT + 5 * loss_dynvisc
        # loss_total = 1 * loss_den +  1 * loss_surfT + 10**4 * loss_kinvisc

        wandb.log({"loss_den": loss_den})
        wandb.log({"loss_visc": loss_dynvisc})
        wandb.log({"loss_surf": loss_surfT})
        # wandb.log({"loss_kinvisc": loss_kinvisc})

        wandb.log({"loss_den_chunk": 5 * loss_dynvisc})
        wandb.log({"loss_visc_chunk": 5 * loss_dynvisc})
        wandb.log({"loss_surf_chunk": 1 * loss_surfT})
        # wandb.log({"loss_kinvisc_chunk": 10**3 * loss_kinvisc})
    """