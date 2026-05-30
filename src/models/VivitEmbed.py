import timm
import torch
import torch.nn as nn

from models.vivit.configuration_vivit import VivitConfig
from models.vivit.modeling_vivit import VivitModel


class VivitEmbed(nn.Module):
    def __init__(
        self,
        dropout,
        output_size,
        class_bool,
        visc_class,
        gmm_num,
        rpm_bool,
        pat_bool,
        num_frames=50,
        image_size=224,
        pat_mode="legacy",
        pattern_gate_init=0.01,
    ):
        super(VivitEmbed, self).__init__()
        if pat_mode not in {"legacy", "embedding", "late_concat", "late_residual"}:
            raise ValueError(f"Unsupported pat_mode: {pat_mode}")
        early_pattern_bool = pat_bool and pat_mode in {"legacy", "embedding"}

        ##### for pretrained model
        # self.config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
        # self.featureextractor = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=self.config)

        self.config = VivitConfig(
            hidden_size=256,  # ViViT-L
            num_hidden_layers=10,  # 20
            num_attention_heads=8,
            intermediate_size=1024,  # 1024
            tubelet_size=(2, 16, 16),
            image_size=int(image_size),
            num_frames=int(num_frames),
            num_channels=3,
            use_mean_pooling=False,
            rpm_bool=rpm_bool,
            pat_bool=early_pattern_bool,
            pat_mode=pat_mode,
        )

        self.featureextractor = VivitModel(self.config)
        self.hidden_size = self.config.hidden_size
        self.pat_bool = pat_bool
        self.pat_mode = pat_mode
        self.pattern_gate = (
            nn.Parameter(torch.tensor(float(pattern_gate_init)))
            if self.pat_bool and self.pat_mode in {"late_concat", "late_residual"}
            else None
        )

        if self.pat_bool and self.pat_mode in {"legacy", "late_concat", "late_residual"}:
            self.pat_backbone = timm.create_model("resnet18", pretrained=True, num_classes=0)
            self.pat_proj = nn.Linear(512, self.hidden_size)

            for p in self.pat_backbone.parameters():
                p.requires_grad = False
            self.pat_backbone.eval()
        else:
            self.pat_backbone = None
            self.pat_proj = None

        # self.pat_embed = nn.Sequential(
        #     nn.Conv2d(3, 256, kernel_size=16, stride=16),  # (B,256,Hp,Wp)
        #     nn.Flatten(2),                                                 # (B,256,N)
        # )

        # FC HEAD
        fc_input_size = self.hidden_size * 2 if self.pat_bool and self.pat_mode == "late_concat" else self.hidden_size
        if class_bool:
            self.fc = nn.Sequential(nn.Linear(fc_input_size, 192), nn.SiLU(), nn.Linear(192, visc_class))
        else:
            self.fc = nn.Sequential(nn.Linear(fc_input_size, 192), nn.SiLU(), nn.Linear(192, 3))

    def _pattern_features(self, pattern):
        pattern = pattern.permute(0, 3, 1, 2).contiguous()
        self.pat_backbone.eval()
        with torch.no_grad():
            features = self.pat_backbone(pattern)
        return self.pat_proj(features)

    def forward(self, video, rpm_idx, pattern):
        outputs = self.featureextractor(video, rpm_idx, pattern)
        video_features = outputs.last_hidden_state.mean(dim=1).contiguous()

        if self.pat_bool and self.pat_mode in {"legacy", "late_concat", "late_residual"}:
            pat_features = self._pattern_features(pattern)
            if self.pat_mode == "legacy":
                video_features = video_features - pat_features
            elif self.pat_mode == "late_concat":
                video_features = torch.cat([video_features, self.pattern_gate * pat_features], dim=1)
            elif self.pat_mode == "late_residual":
                video_features = video_features - self.pattern_gate * pat_features

        viscosity = self.fc(video_features)

        return viscosity
