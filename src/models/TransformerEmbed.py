import torch
import torch.nn as nn
from transformers import VivitModel, VivitConfig, VivitForVideoClassification

class TransformerEmbed(nn.Module):
    def __init__(self, dropout, output_size, flow_bool):
        super(TransformerEmbed, self).__init__()
        # self.config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
        # self.featureextractor = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=self.config)

        # self.config = VivitConfig()

        self.config = VivitConfig(
            hidden_size=1024,              # ViViT-L
            num_hidden_layers=20,
            num_attention_heads=16,
            intermediate_size=4096,
            tubelet_size=(2, 16, 16),
            image_size=224,
            num_frames=32,
            num_channels=3,
            use_mean_pooling=False
        )

        self.featureextractor = VivitModel(self.config)
        self.hidden_size = self.config.hidden_size
        # self.rpm_embedding = nn.Embedding(50, self.hidden_size)

        # FC HEAD
        self.flow_bool = flow_bool

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 192),
            nn.SiLU(),
            nn.Linear(192, 10)
        )

    def forward(self, video: torch.Tensor, rpm: torch.Tensor):
        outputs = self.featureextractor(video)
        video_features = outputs.last_hidden_state.mean(dim=1).contiguous()
        # rpm_vec = self.rpm_embedding(torch.round(rpm).squeeze(-1).long())

        # concat = video_features + rpm_vec
        concat = video_features

        if self.flow_bool:
            viscosity = concat
        else:
            # viscosity = self.fc(concat)
            viscosity = self.classifier(concat)

        return viscosity