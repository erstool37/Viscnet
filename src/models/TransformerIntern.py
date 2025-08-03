import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TransformerIntern(nn.Module):
    def __init__(self, dropout, output_size, flow_bool):
        super(TransformerIntern, self).__init__()
        model_name = "OpenGVLab/InternVideo-MM-B-16-224px"

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_frames = 32
        config.use_mean_pooling = True

        self.featureextractor = AutoModel.from_pretrained(
            model_name, config=config, trust_remote_code=True
        )
        self.hidden_size = self.featureextractor.config.hidden_size  # usually 768 for B models

        self.rpm_embedding = nn.Embedding(50, self.hidden_size)
        self.flow_bool = flow_bool

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 192),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(192, output_size)
        )

    def forward(self, video: torch.Tensor, rpm: torch.Tensor):
        outputs = self.featureextractor(pixel_values=video)
        video_features = outputs.last_hidden_state.mean(dim=1)
        rpm_vec = self.rpm_embedding(torch.round(rpm).squeeze(-1).long())
        concat = video_features + rpm_vec

        return concat if self.flow_bool else self.classifier(concat)