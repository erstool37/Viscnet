import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vivit.modeling_vivit import VivitModel
from models.vivit.configuration_vivit import VivitConfig

# ----------------------------- RQS (1D) -----------------------------

def _rqs_1d(y, widths, heights, derivatives, tail_bound=3.0, inverse=False, eps=1e-6):
    """
    y: [B,1]; widths, heights: [B,K]; derivatives: [B,K+1]
    Returns: (x_or_y, log|dy/dx| or log|dx/dy|) both [B,1]
    """
    B, K = y.size(0), widths.size(1)

    # Constrain parameters
    widths  = F.softmax(widths,  dim=-1)          # partition of unity
    heights = F.softmax(heights, dim=-1)
    derivatives = F.softplus(derivatives) + 1e-3  # positive, bounded away from 0

    # Knot positions in [-T, T]
    xk = torch.cumsum(torch.cat([torch.zeros(B,1, device=y.device), widths], dim=-1), dim=-1)
    yk = torch.cumsum(torch.cat([torch.zeros(B,1, device=y.device), heights], dim=-1), dim=-1)
    xk = 2*tail_bound*(xk - 0.5)   # [B,K+1]
    yk = 2*tail_bound*(yk - 0.5)

    # Clamp y just inside the domain (avoid exact boundary ambiguity)
    y = y.clamp(-tail_bound + 1e-4, tail_bound - 1e-4)

    # Bin index: count how many left-knots are <= y, then subtract 1, clamp to [0,K-1]
    # xk_left = xk[:, :-1]  # [B,K]
    idx = (y >= xk[:, :-1]).sum(dim=1) - 1
    idx = idx.clamp(min=0, max=K-1)  # [B,1]

    def G(M):  # gather along last dim using scalar idx per batch
        return M.gather(1, idx)

    xL, xR = G(xk[:, :-1]), G(xk[:, 1:])
    yL, yR = G(yk[:, :-1]), G(yk[:, 1:])
    dL, dR = G(derivatives[:, :-1]), G(derivatives[:, 1:])

    w = (xR - xL) + eps
    h = (yR - yL) + eps
    s = h / w

    if not inverse:  # forward: x -> y
        t = (y - xL) / w
        num = h*(s*t*t + dL*t*(1 - t))
        den = s + (dR + dL - 2*s)*t*(1 - t)
        y_out = yL + num / (den + eps)

        dydx = (s*s * (dR*t*t + 2*s*t*(1 - t) + dL*(1 - t)*(1 - t))) / (den*den + eps)
        logabsdet = torch.log(dydx + eps)
        return y_out, logabsdet

    else:         # inverse: y -> x (Newton iterations in t-space)
        y_target = y
        t = ((y_target - yL) / h).clamp(1e-4, 1-1e-4)
        for _ in range(10):
            num = h*(s*t*t + dL*t*(1 - t))
            den = s + (dR + dL - 2*s)*t*(1 - t)
            y_curr = yL + num / (den + eps)

            dnum = h*(2*s*t + dL*(1 - 2*t))
            dden = (dR + dL - 2*s)*(1 - 2*t)
            dydt = (dnum*den - num*dden) / (den*den + eps)

            t = (t - (y_curr - y_target) / (dydt + eps)).clamp(1e-4, 1-1e-4)

        x_out = xL + w*t
        # same dydx expression but evaluated at final t
        dydx = (s*s * (dR*t*t + 2*s*t*(1 - t) + dL*(1 - t)*(1 - t))) / (den*den + eps)
        logabsdet = -torch.log(dydx + eps)  # log|dx/dy|
        return x_out, logabsdet


class ConditionalSpline1D(nn.Module):
    """
    A stack of L conditional 1D spline transforms.
    Each layer conditions on [x,h] to output K-bin parameters.
    """
    def __init__(self, h_dim, K_bins=16, L_layers=4, hidden=256, tail_bound=3.0):
        super().__init__()
        self.K, self.L, self.T = K_bins, L_layers, tail_bound
        self.out_dim = 3*self.K + 1  # widths(K), heights(K), derivatives(K+1)
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim + 1, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, self.out_dim)
            ) for _ in range(self.L)
        ])

    def forward(self, y, h, inverse=False):
        """
        y: [B,1], h: [B,h_dim]
        returns: (out, sum_logdet) both [B,1]
        """
        x = y
        sum_logdet = torch.zeros_like(y)
        if not inverse:          # x->y
            for net in self.nets:
                params = net(torch.cat([x, h], dim=1))
                w, he, d = torch.split(params, [self.K, self.K, self.K+1], dim=-1)
                x, logdet = _rqs_1d(x, w, he, d, tail_bound=self.T, inverse=False)
                sum_logdet = sum_logdet + logdet
            return x, sum_logdet
        else:                    # y->x
            for net in reversed(self.nets):
                params = net(torch.cat([x, h], dim=1))
                w, he, d = torch.split(params, [self.K, self.K, self.K+1], dim=-1)
                x, logdet = _rqs_1d(x, w, he, d, tail_bound=self.T, inverse=True)
                sum_logdet = sum_logdet + logdet
            return x, sum_logdet

# --------------------------- ViViT + CNF (1D) ---------------------------

class VivitCNF(nn.Module):
    """
    Trains a 1-D conditional flow for the MIDDLE target only (y_mid = y[:,1:2]).
    Provides: .nll(...) for training, .sample(...) for inference, and an optional fc head.
    """
    def __init__(self, dropout, output_size, class_bool, visc_class):
        super(VivitCNF, self).__init__()
        self.config = VivitConfig(
            hidden_size=256,
            num_hidden_layers=10,
            num_attention_heads=8,
            intermediate_size=256,
            tubelet_size=(2, 16, 16),
            image_size=224,
            num_frames=32,
            num_channels=3,
            use_mean_pooling=False,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.featureextractor = VivitModel(self.config)
        d = self.config.hidden_size

        # Project pooled ViViT features → conditioner h
        self.proj = nn.Sequential(
            nn.Linear(d, 192), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(192, 128)
        )
        self.h_dim = 128

        # 1D conditional flow for the middle component
        self.flow = ConditionalSpline1D(h_dim=self.h_dim, K_bins=16, L_layers=4, hidden=256, tail_bound=3.0)

        # Optional legacy head (now sized to 128 to match proj output)
        if class_bool:
            self.fc = nn.Sequential(nn.Linear(128, 192), nn.SiLU(), nn.Linear(192, visc_class))
        else:
            self.fc = nn.Sequential(nn.Linear(128, 192), nn.SiLU(), nn.Linear(192, 3))

        # Standardization (scalar stats for the single modeled dim)
        self.register_buffer("y_mean", torch.zeros(1,1))  # [1,1]
        self.register_buffer("y_std",  torch.ones(1,1))   # [1,1]
        self.LOG2PI = torch.log(torch.tensor(2.0*3.141592653589793))

    # ---- utilities ----
    def _pool_feats(self, video, rpm_idx):
        out = self.featureextractor(video, rpm_idx)              # last_hidden_state [B,N,d]
        feats = out.last_hidden_state.mean(dim=1).contiguous()   # [B,d]
        h = self.proj(feats)                                     # [B,128]
        return h

    def _standardize(self, y):
        return (y - self.y_mean) / (self.y_std + 1e-8)

    def _destandardize(self, y_std):
        return y_std * (self.y_std + 1e-8) + self.y_mean

    # ---- TRAIN: Negative log-likelihood for y_mid ----
    def nll(self, video, rpm_idx, y_all):
        """
        y_all: [B,3]; this will use only y_mid = y_all[:,1:2]
        returns: scalar NLL
        """
        h = self._pool_feats(video, rpm_idx)          # [B,128]
        y_mid = y_all[:, 1:2]                         # [B,1]
        y_std = self._standardize(y_mid)              # [B,1]

        # data -> base
        z, logdet = self.flow(y_std, h, inverse=True) # z ~ N(0,1); both [B,1]
        log_pz = -0.5 * (z*z + self.LOG2PI)           # [B,1]
        nll = -(log_pz + logdet).mean()               # scalar
        return nll

    # ---- INFERENCE: sample y_mid given (video, rpm) ----
    @torch.no_grad()
    def sample(self, video, rpm_idx, num_samples=1):
        """
        Returns: y_mid samples of shape [num_samples, B, 1]
        """
        h = self._pool_feats(video, rpm_idx)          # [B,128]
        B = h.size(0)
        outs = []
        for _ in range(num_samples):
            z = torch.randn(B, 1, device=h.device)
            y_std, _ = self.flow(z, h, inverse=False) # base -> data
            outs.append(self._destandardize(y_std))
        return torch.stack(outs, dim=0)

    # ---- optional: legacy head on h (NOT used by CNF) ----
    def feat_head(self, video, rpm_idx):
        h = self._pool_feats(video, rpm_idx)
        return self.fc(h)

    # Keep a forward for compatibility if you want
    def forward(self, video, rpm_idx, y=None, mode="nll"):
        if mode == "nll":
            assert y is not None, "Pass y [B,3]; the middle dim will be used."
            return self.nll(video, rpm_idx, y)
        elif mode == "sample":
            return self.sample(video, rpm_idx, num_samples=1).squeeze(0)  # [B,1]
        elif mode == "feat_head":
            return self.feat_head(video, rpm_idx)
        else:
            raise ValueError("mode must be 'nll' | 'sample' | 'feat_head'")