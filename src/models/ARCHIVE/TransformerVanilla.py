import torch
import torch.nn as nn
from transformers import VivitModel, VivitConfig

class TransformerVanilla(nn.Module):
    """
    does not have rpm embedding.
    """
    def __init__(self, dropout, output_size, flow_bool):
        super(TransformerVanilla, self).__init__()
        self.config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.featureextractor = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=self.config)
        self.hidden_size = self.config.hidden_size  # Usually 768 for ViViT-base

        # FC HEAD
        self.flow_bool = flow_bool
        # self.fc = nn.Sequential(
        #     nn.Linear(self.hidden_size * 2, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),

        #     nn.Linear(512, 96),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),

        #     nn.Linear(96, 12),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),

        #     nn.Linear(12, output_size),
        # )

        self.rpm_embedding = nn.Sequential(
            nn.Linear(1, 96),
            nn.ReLU(),
            nn.Linear(96, self.hidden_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 5),
        )

    def forward(self, video: torch.Tensor, rpm: torch.Tensor):
        """
        video: (B, T, C, H, W)
        """
        # ViViT expects (B, T, C, H, W)
        outputs = self.featureextractor(video)
        video_features = outputs.pooler_output
        rpm_vec = self.rpm_embedding(rpm.unsqueeze(1))
        
        # concat = video_features
        concat = torch.cat((video_features, rpm_vec), dim=-1).contiguous()

        if self.flow_bool:
            viscosity = concat
        else:
            # viscosity = self.fc(concat)
            viscosity = self.classifier(concat)

        return viscosity