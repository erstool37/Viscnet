import math
import torch
import torch.nn as nn
import wandb

class NLL(nn.Module):
    def __init__(self, unnormalizer=None, path=None):
        super().__init__()

    def forward(self, z, log_det_jacobian):
        # z: [B, D], log_det_jacobian: [B]
        B, D = z.shape
        quad = 0.5 * (z**2).sum(dim=1)                  # [B]
        const = 0.5 * D * math.log(2.0 * math.pi)       # scalar
        nll = quad + const - log_det_jacobian           # [B]
        loss_total = nll.mean()

        # Optional diagnostics (per-dim quadratic terms)
        per_dim_quad = 0.5 * (z**2).mean(dim=0)         # [D]
        wandb.log({
            "loss_total": loss_total,
            "quad_mean": quad.mean(),
            "neg_logdet_mean": (-log_det_jacobian).mean(),
        })
        # If D==3 and you want named logs:
        if z.shape[1] >= 3:
            wandb.log({
                "quad_den": per_dim_quad[0],
                "quad_visc": per_dim_quad[1],
                "quad_surfT": per_dim_quad[2],
            })
        return loss_total