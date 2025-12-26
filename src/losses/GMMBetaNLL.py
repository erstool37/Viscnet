import torch
import torch.nn as nn
import math

class GMMBetaNLL(nn.Module):
    """
    True β-NLL for Gaussian Mixture regression.
    Derived from:
    'On the Pitfalls of Heteroscedastic Uncertainty Estimation 
     with Probabilistic Neural Networks'
    """

    def __init__(self, descaler, data_root_train, smoothing_label, beta: float = 1.0, eps: float = 1e-9):
        super().__init__()
        self.beta = beta
        self.eps = eps

    @staticmethod
    def _extract_target(parameters: torch.Tensor):
        return parameters[:, 2].unsqueeze(-1)  # [B,1]

    def forward(self, outputs: dict, parameters: torch.Tensor):
        """
        outputs:
            pi:    [B,K]
            mu:    [B,K]
            sigma: [B,K]  (std)

        returns:
            scalar β-NLL of the true GMM
        """
        pi    = outputs["pi"]        # [B,K]
        mu    = outputs["mu"]        # [B,K]
        sigma = outputs["sigma"]     # [B,K]
        y     = self._extract_target(parameters)  # [B,1]

        # component variance
        var = (sigma ** 2)  # [B,K]

        # --- β applied ONLY to squared error (true β-NLL) ---
        # log N_beta(y | mu, sigma):
        log_prob = -0.5 * (torch.log(2 * math.pi * var) + (y - mu)**2 / (var ** self.beta))  # [B,K]

        # mixture weighting in log space
        log_pi = torch.log(pi)  # [B,K]

        # stable mixture log likelihood
        log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # [B]

        # final NLL
        return -(log_mix.mean())