import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vivit.modeling_vivit import VivitModel
from models.vivit.configuration_vivit import VivitConfig

class VivitGMM(nn.Module):
    def __init__(self, dropout, output_size, class_bool, visc_class, gmm_num):
        super(VivitGMM, self).__init__()

        self.config = VivitConfig(
            hidden_size=256,
            num_hidden_layers=20,
            num_attention_heads=8,
            intermediate_size=1024,
            tubelet_size=(2, 16, 16),
            image_size=224,
            num_frames=50,
            num_channels=3,
            use_mean_pooling=False
        )   

        self.featureextractor = VivitModel(self.config)
        # for param in self.featureextractor.parameters():
        #     param.requires_grad = False
        
        H = self.config.hidden_size
        # K = visc_class if isinstance(visc_class, int) and visc_class > 1 else 5
        self.K = gmm_num

        # Simple GMM head: predict (pi, mu, sigma) for scalar y
        self.gmm_pi   = nn.Linear(H, self.K)
        self.gmm_mu   = nn.Linear(H, self.K)
        self.gmm_lsig = nn.Linear(H, self.K)

    def forward(self, video, rpm_idx):
        outputs = self.featureextractor(video, rpm_idx)
        z = outputs.last_hidden_state.mean(dim=1).contiguous()

        # GMM layer
        pi = F.softmax(self.gmm_pi(z), dim=-1)                  # [B,K]
        mu = self.gmm_mu(z)                                     # [B,K]
        sigma = F.softplus(self.gmm_lsig(z))       # [B,K]

        # Mixture mean (point estimate)
        y_hat = (pi * mu).sum(-1, keepdim=True)                 # [B,1]

        return {"pi": pi, "mu": mu, "sigma": sigma, "y_hat": y_hat}