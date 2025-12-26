import math
import torch
import torch.nn as nn

class GMMNLL(nn.Module):
    """
    Negative log-likelihood for a scalar GMM head, where the target tensor
    has 3 features and viscosity is at index 1.
    Expects model outputs: {"pi":[B,K], "mu":[B,K], "sigma":[B,K], "y_hat":[B,1]}
    """
    def __init__(self, descaler, data_root_train, smoothing_label):
        super().__init__()
        self.eps = 1e-12

    @staticmethod
    def _extract_viscosity(parameters: torch.Tensor) -> torch.Tensor:
        """
        parameters: [..., 3]; viscosity is feature at index 1
        Returns shape [B] (flattening any leading dims into batch).
        """
        
        y = parameters[:, 2]                # pick viscosity
        if y.dim() == 1:
            return y
        # Flatten all but last dim to batch
        return y.reshape(-1)

    def forward(self, outputs: dict, parameters: torch.Tensor) -> torch.Tensor:
        """
        outputs: dict with pi, mu, sigma, y_hat
        parameters: tensor with last-dim size 3; viscosity at index 1
        Returns scalar mean NLL over batch.
        """
        pi    = outputs["pi"]      # [B,K]
        mu    = outputs["mu"]      # [B,K]
        sigma = outputs["sigma"]   # [B,K]

        y = self._extract_viscosity(parameters)   # [B]
        if y.dim() == 1:
            y = y.unsqueeze(-1)                   # [B,1]

        var = sigma ** 2                          # [B,K]
        # log N(y; mu, var)
        log_prob = -0.5 * (torch.log(2 * math.pi * var) + (y - mu) ** 2 / var)  # [B,K]
        # log sum_k pi_k * N_k
        log_mix = torch.logsumexp(torch.log(pi) + log_prob, dim=-1)  # [B]
        nll = -(log_mix).mean()                                                   # scalar

        return nll

# import torch
# import torch.nn as nn
# import math
# import torch.nn.functional as F

# class GMMNLL(nn.Module):
#     """
#     β-NLL for scalar Gaussian Mixture regression.
#     1. Collapse mixture via law of total variance
#     2. Apply β reweighting to down/up-weight high uncertainty

#     arguments:
#     beta ∈ [0,1]
#         β = 0   → strong emphasis on low σ (trust confident samples)
#         β = 1   → equal weighting for all samples
#     """

#     def __init__(self, unnormalizer, path, smooth_label, beta: float = 0.10, eps: float = 1e-9):
#         super().__init__()
#         self.beta = beta
#         self.eps = eps

#     @staticmethod
#     def _extract_target(parameters: torch.Tensor) -> torch.Tensor:
#         """
#         viscosity is at index 2 in your tensor: [..., 3]
#         returns [B,1]
#         """
#         t = parameters[:, 2].unsqueeze(-1)
#         return t

#     def forward(self, outputs: dict, parameters: torch.Tensor):
#         """
#         outputs:
#             pi:    [B,K]
#             mu:    [B,K]
#             sigma: [B,K]
#         parameters: [...,3]; viscosity at idx 2
#         returns: scalar β-NLL
#         """
#         pi    = outputs["pi"]        # [B,K]
#         mu    = outputs["mu"]        # [B,K]
#         sigma = outputs["sigma"]     # [B,K]

#         target = self._extract_target(parameters)  # [B,1]

#         # --- Mixture collapsed mean ---
#         mu_mix = (pi * mu).sum(dim=-1, keepdim=True)  # [B,1]

#         # --- Mixture collapsed variance (law of total variance) ---
#         # Var = E[var] + E[mu^2] - (E[mu])^2
#         comp = sigma**2 + mu**2                       # [B,K]
#         var_mix = (pi * comp).sum(dim=-1, keepdim=True) - mu_mix**2  # [B,1]
#         var_mix = var_mix.clamp(min=self.eps)

#         # --- Core Gaussian NLL ---
        
#         nll = 0.5 * ((target - mu_mix)**2 / var_mix + torch.log(var_mix))



#         # --- β reweighting ---
#         if self.beta > 0:
#             nll = nll * (var_mix.detach() ** self.beta)
#         # mse = F.mse_loss(mu_mix, target)

#         return nll.mean()
#         # return mse