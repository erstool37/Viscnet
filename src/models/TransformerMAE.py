import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TransformerMAE(nn.Module):
    def __init__(self, dropout, output_size, flow_bool):
        super(TransformerMAE, self).__init__()
        config = AutoConfig.from_pretrained(
            "OpenGVLab/VideoMAEv2-Large",
            trust_remote_code=True,
            from_tf=True
        )
        config.use_mean_pooling = True  # ensure pooled output

        self.featureextractor = AutoModel.from_pretrained(
            "OpenGVLab/VideoMAEv2-Large",
            config=config,
            trust_remote_code=True
        )
        self.hidden_size = 1024  # fixed hidden size for VideoMAEv2-Large

        # self.rpm_embedding = nn.Embedding(50, self.hidden_size)
        self.rpm_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
        )

        self.flow_bool = flow_bool

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 192),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(192, 10)
        )

    def forward(self, video: torch.Tensor, rpm: torch.Tensor):
        video_features = self.featureextractor(video)
        rpm_idx = rpm.float().unsqueeze(-1)  # shape: (B, 1)
        rpm_vec = self.rpm_embedding(rpm_idx)  # shape: (B, hidden_size)
        # rpm_vec = self.rpm_embedding(torch.round(rpm).squeeze(-1).long())
        concat = video_features + rpm_vec
        
        return concat if self.flow_bool else self.classifier(concat)