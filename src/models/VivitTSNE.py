import torch
import torch.nn as nn
from models.vivit.modeling_vivit import VivitModel
from models.vivit.configuration_vivit import VivitConfig

class VivitTSNE(nn.Module):
    def __init__(self, dropout, output_size, class_bool, visc_class):
        super(VivitTSNE, self).__init__()

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
            use_mean_pooling=False
        )

        self.featureextractor = VivitModel(self.config)
        self.hidden_size = self.config.hidden_size

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

    def forward(self, video, rpm_idx):
        outputs = self.featureextractor(video, rpm_idx)
        video_features = outputs.last_hidden_state.mean(dim=1).contiguous()
        viscosity = self.fc(video_features)

        return viscosity, video_features