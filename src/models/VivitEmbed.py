import torch
import torch.nn as nn
from models.vivit.modeling_vivit import VivitModel
from models.vivit.configuration_vivit import VivitConfig
import timm

class VivitEmbed(nn.Module):
    def __init__(self, dropout, output_size, class_bool, visc_class, gmm_num, rpm_bool, pat_bool):
        super(VivitEmbed, self).__init__()

        ##### for pretrained model
        # self.config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
        # self.featureextractor = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=self.config)

        self.config = VivitConfig(
            hidden_size=256,              # ViViT-L
            num_hidden_layers=10, #20
            num_attention_heads=8,
            intermediate_size=1024, #1024
            tubelet_size=(2, 16, 16),
            image_size=224,
            num_frames=50,
            num_channels=3,
            use_mean_pooling=False,
            rpm_bool=rpm_bool,
            pat_bool=pat_bool
        )

        self.featureextractor = VivitModel(self.config)
        self.hidden_size = self.config.hidden_size


        # pattern related
        self.pat_backbone = timm.create_model("resnet18", pretrained=True, num_classes=0)  # outputs (B,512)
        self.pat_proj = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.LayerNorm(512),
            # nn.GELU(),
            nn.Linear(512, 256))

        for p in self.pat_backbone.parameters():
            p.requires_grad = False
        self.pat_backbone.eval()  # (keep BN fixed)

        # self.pat_embed = nn.Sequential(
        #     nn.Conv2d(3, 256, kernel_size=16, stride=16),  # (B,256,Hp,Wp)
        #     nn.Flatten(2),                                                 # (B,256,N)
        # )

        # FC HEAD
        if class_bool:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size, 192),
                nn.SiLU(),
                nn.Linear(192, visc_class))
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size, 192),
                nn.SiLU(),
                nn.Linear(192, 3))

    def forward(self, video, rpm_idx, pattern):
        outputs = self.featureextractor(video, rpm_idx, pattern)
        video_features = outputs.last_hidden_state.mean(dim=1).contiguous()
        
        pattern = pattern.permute(0, 3, 1, 2).contiguous()
        pat_features = self.pat_proj(self.pat_backbone(pattern))
        total_features = video_features - pat_features

        # viscosity = self.fc(video_features)
        viscosity = self.fc(total_features)

        return viscosity