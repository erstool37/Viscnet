import torch
import torch.nn as nn
from transformers import VivitModel, VivitConfig

class TransformerEmbed(nn.Module):
    def __init__(self, dropout, output_size, flow_bool):
        super(TransformerEmbed, self).__init__()
        self.config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.featureextractor = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=self.config)
        self.hidden_size = self.config.hidden_size

        # RPM EMBEDDING
        # self.rpm_embedding = nn.Sequential(
        #     nn.Linear(1, 96),
        #     nn.ReLU(),
        #     nn.Linear(96, self.hidden_size),
        # )
        self.rpm_embedding = nn.Embedding(10, self.hidden_size)

        # FC HEAD
        self.flow_bool = flow_bool

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 192),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(192,10)

        )

    def forward(self, video: torch.Tensor, rpm: torch.Tensor):
        outputs = self.featureextractor(video)
        video_features = outputs.pooler_output
        rpm_vec = self.rpm_embedding(torch.round(rpm).squeeze(-1).long())

        concat = video_features + rpm_vec

        if self.flow_bool:
            viscosity = concat
        else:
            # viscosity = self.fc(concat)
            viscosity = self.classifier(concat)

        return viscosity

"""
        # VideoMAE encoder
        self.config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
        self.config.model_config["use_mean_pooling"] = False
        self.featureextractor = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Base', config=self.config, trust_remote_code=True)
        self.hidden_size = self.config.model_config["embed_dim"]
        self.tube_frame = self.config.model_config["tubelet_size"]
        self.patch_size = self.config.model_config["patch_size"]

        # RPM EMBEDDING
        self.rpm_embedding = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # FC HEAD
        self.flow_bool = flow_bool
        self.fc =nn.Sequential(
            nn.Linear(self.hidden_size * 2, 192), # 768 * 2
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(192, 24),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(24, output_size),
        )

    def forward(self, video:torch.Tensor, rpm:torch.Tensor):
        video: (B, 3, T, H, W)
        # B, C, T, H, W = video.shape
        
        video_features = self.featureextractor(video)
    
        rpm_vec = self.rpm_embedding(rpm.unsqueeze(1))
    
        concat = torch.cat((video_features, rpm_vec), dim=-1)

        if self.flow_bool:
            viscosity = concat
        else:
            viscosity = self.fc(concat)
            
        return viscosity
    
# config structure for videoMAE v2 Base

{
    "_name_or_path": "./",
    "model_type": "VideoMAEv2_Base",
    "architectures": [
      "VideoMAEv2_Base"
    ],
    "auto_map": {
        "AutoModel": "modeling_videomaev2.VideoMAEv2",
        "AutoConfig": "modeling_config.VideoMAEv2Config"
    },
    "model_config":{
      "img_size": 224,
      "patch_size": 16,
      "in_chans": 3,
      "num_classes": 0,
      "embed_dim": 768,
      "depth": 12,
      "num_heads": 12,
      "mlp_ratio": 4,
      "qkv_bias": true,
      "qk_scale": null,
      "drop_rate": 0.0,
      "attn_drop_rate": 0.0,
      "drop_path_rate": 0.0,
      "norm_layer": "nn.LayerNorm",
      "layer_norm_eps": 1e-6,
      "init_values": 0.0,
      "use_learnable_pos_emb": false,
      "tubelet_size": 2,
      "use_mean_pooling": true,
      "with_cp": false,
      "num_frames": 16,  
      "cos_attn": false
    },
    "transformers_version": "4.38.0",
    "use_cache": true
  }
"""