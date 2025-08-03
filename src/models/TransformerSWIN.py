import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification

class TransformerSWIN(nn.Module):
    def __init__(self, dropout, output_size, flow_bool):
        super(TransformerSWIN, self).__init__()
        self.featureextractor = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
        self.hidden_size = self.featureextractor.config.hidden_size  # typically 768

        self.rpm_embedding = nn.Embedding(50, self.hidden_size)
        self.flow_bool = flow_bool

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 192),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(192, 10)
        )

    def forward(self, video: torch.Tensor, rpm: torch.Tensor):
        # input shape: (B, C, T, H, W)
        outputs = self.featureextractor(pixel_values=video)
        video_features = outputs.logits
        rpm_vec = self.rpm_embedding(torch.round(rpm).squeeze(-1).long())
        concat = video_features + rpm_vec

        if self.flow_bool:
            return concat
        else:
            return self.classifier(concat)