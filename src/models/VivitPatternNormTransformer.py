import torch
import torch.nn as nn

from models.vivit.configuration_vivit import VivitConfig
from models.vivit.modeling_vivit import VivitEncoder, VivitTubeletEmbeddings


class PatternVideoCrossAttention(nn.Module):
    """Fuse video tokens with spatial clean-pattern tokens through cross-attention."""

    def __init__(self, dim, num_heads, dropout=0.0, use_diff=True, use_product=True):
        super().__init__()
        self.use_diff = bool(use_diff)
        self.use_product = bool(use_product)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        feature_count = 2 + int(self.use_diff) + int(self.use_product)
        self.fusion = nn.Linear(feature_count * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, video_tokens, pattern_tokens, need_weights=False):
        cross, attention_weights = self.attention(
            query=video_tokens,
            key=pattern_tokens,
            value=pattern_tokens,
            need_weights=need_weights,
        )
        features = [video_tokens, cross]
        if self.use_diff:
            features.append(video_tokens - cross)
        if self.use_product:
            features.append(video_tokens * cross)
        normalized = self.fusion(torch.cat(features, dim=-1))
        output = self.norm(video_tokens + self.dropout(normalized))
        if need_weights:
            return output, attention_weights
        return output


class PatternPatchEmbeddings(nn.Module):
    """Patchify a clean background pattern without global pooling."""

    def __init__(self, image_size, patch_size, num_channels, hidden_size):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, pattern):
        tokens = self.projection(pattern).flatten(2).transpose(1, 2)
        return tokens


class VivitPatternNormTransformer(nn.Module):
    """Reference-conditioned ViViT that compares video tokens to clean pattern tokens."""

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
        pattern_norm=None,
    ):
        super().__init__()
        del gmm_num, pat_mode, pattern_gate_init
        pattern_norm = dict(pattern_norm or {})
        self.class_bool = bool(class_bool)
        self.output_size = int(output_size)
        self.visc_class = int(visc_class)
        self.rpm_bool = bool(rpm_bool)
        self.pat_bool = bool(pat_bool)
        self.store_cross_attention = bool(pattern_norm.get("store_cross_attention", False))
        self.last_cross_attention = None

        hidden_size = int(pattern_norm.get("hidden_size", 256))
        num_heads = int(pattern_norm.get("num_attention_heads", 8))
        intermediate_size = int(pattern_norm.get("intermediate_size", hidden_size * 4))
        num_hidden_layers = int(pattern_norm.get("num_hidden_layers", 10))
        tubelet_size = tuple(pattern_norm.get("tubelet_size", (2, 16, 16)))
        pattern_patch_size = int(pattern_norm.get("pattern_patch_size", tubelet_size[1]))
        pattern_encoder_layers = int(pattern_norm.get("pattern_encoder_layers", 1))
        video_encoder_layers = int(pattern_norm.get("video_encoder_layers", 0))
        cross_attention_layers = int(pattern_norm.get("cross_attention_layers", 1))
        cross_heads = int(pattern_norm.get("num_cross_attention_heads", num_heads))
        self.add_rpm_before_cross_attention = bool(pattern_norm.get("add_rpm_before_cross_attention", False))
        self.add_rpm_after_cross_attention = bool(pattern_norm.get("add_rpm_after_cross_attention", True))

        self.config = VivitConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            tubelet_size=tubelet_size,
            image_size=int(image_size),
            num_frames=int(num_frames),
            num_channels=3,
            use_mean_pooling=False,
            rpm_bool=False,
            pat_bool=False,
        )
        self.hidden_size = self.config.hidden_size
        self.video_patch_embeddings = VivitTubeletEmbeddings(self.config)
        self.pattern_patch_embeddings = PatternPatchEmbeddings(
            image_size=image_size,
            patch_size=pattern_patch_size,
            num_channels=self.config.num_channels,
            hidden_size=self.hidden_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.video_position_embeddings = nn.Parameter(
            torch.zeros(1, self.video_patch_embeddings.num_patches + 1, self.hidden_size)
        )
        self.pattern_position_embeddings = nn.Parameter(
            torch.zeros(1, self.pattern_patch_embeddings.num_patches, self.hidden_size)
        )
        self.rpm_embed = nn.Embedding(self.config.rpm_class, self.hidden_size)
        self.dropout = nn.Dropout(float(dropout))

        self.pattern_encoder = self._make_token_encoder(
            pattern_encoder_layers,
            self.hidden_size,
            num_heads,
            intermediate_size,
            float(dropout),
        )
        self.video_encoder = self._make_token_encoder(
            video_encoder_layers,
            self.hidden_size,
            num_heads,
            intermediate_size,
            float(dropout),
        )
        self.cross_attention = nn.ModuleList(
            [
                PatternVideoCrossAttention(
                    dim=self.hidden_size,
                    num_heads=cross_heads,
                    dropout=float(dropout),
                    use_diff=bool(pattern_norm.get("use_difference_fusion", True)),
                    use_product=bool(pattern_norm.get("use_product_fusion", True)),
                )
                for _ in range(cross_attention_layers)
            ]
        )
        self.encoder = VivitEncoder(self.config)
        self.layernorm = nn.LayerNorm(self.hidden_size, eps=self.config.layer_norm_eps)

        if self.class_bool:
            self.fc = nn.Sequential(nn.Linear(self.hidden_size, 192), nn.SiLU(), nn.Linear(192, self.visc_class))
        else:
            self.fc = nn.Sequential(nn.Linear(self.hidden_size, 192), nn.SiLU(), nn.Linear(192, self.output_size))

        self.apply(self._init_weights)
        self._init_direct_parameters()

    @staticmethod
    def _make_token_encoder(num_layers, hidden_size, num_heads, intermediate_size, dropout):
        if int(num_layers) <= 0:
            return nn.Identity()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _init_direct_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(self.video_position_embeddings, mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(self.pattern_position_embeddings, mean=0.0, std=self.config.initializer_range)

    def _pattern_to_nchw(self, pattern):
        if pattern is None:
            raise ValueError("VivitPatternNormTransformer requires a clean background pattern tensor")
        if pattern.ndim != 4:
            raise ValueError(f"Expected pattern tensor rank 4, got shape {tuple(pattern.shape)}")
        if pattern.shape[1] == self.config.num_channels:
            pattern = pattern.contiguous()
        elif pattern.shape[-1] == self.config.num_channels:
            pattern = pattern.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"Expected pattern channels on dim 1 or -1, got shape {tuple(pattern.shape)}")

        _, _, height, width = pattern.shape
        target = int(self.config.image_size)
        if height >= target and width >= target:
            top = (height - target) // 2
            left = (width - target) // 2
            pattern = pattern[:, :, top : top + target, left : left + target]
        if pattern.shape[-2:] != (target, target):
            pattern = nn.functional.interpolate(pattern, size=(target, target), mode="bilinear", align_corners=False)
        return pattern

    def _rpm_tokens(self, rpm_idx, token_count):
        if rpm_idx is None:
            rpm_idx = torch.zeros(token_count, dtype=torch.long, device=self.cls_token.device)
        rpm_idx = rpm_idx.view(-1).long()
        return self.rpm_embed(rpm_idx).unsqueeze(1)

    def tokenize_video(self, video, rpm_idx=None):
        video_tokens = self.video_patch_embeddings(video)
        video_tokens = video_tokens + self.video_position_embeddings[:, 1:, :]
        if self.rpm_bool and self.add_rpm_before_cross_attention:
            video_tokens = video_tokens + self._rpm_tokens(rpm_idx, video.shape[0])
        video_tokens = self.dropout(video_tokens)
        return self.video_encoder(video_tokens)

    def tokenize_pattern(self, pattern):
        pattern = self._pattern_to_nchw(pattern)
        pattern_tokens = self.pattern_patch_embeddings(pattern)
        pattern_tokens = pattern_tokens + self.pattern_position_embeddings
        pattern_tokens = self.dropout(pattern_tokens)
        return self.pattern_encoder(pattern_tokens)

    def pattern_video_cross_attention(self, video_tokens, pattern_tokens):
        self.last_cross_attention = None
        for layer in self.cross_attention:
            if self.store_cross_attention:
                video_tokens, attention_weights = layer(video_tokens, pattern_tokens, need_weights=True)
                self.last_cross_attention = attention_weights.detach().cpu()
            else:
                video_tokens = layer(video_tokens, pattern_tokens)
        return video_tokens

    def get_last_cross_attention(self):
        return self.last_cross_attention

    def forward(self, video, rpm_idx, pattern):
        video_tokens = self.tokenize_video(video, rpm_idx=rpm_idx)
        pattern_tokens = self.tokenize_pattern(pattern)
        video_tokens = self.pattern_video_cross_attention(video_tokens, pattern_tokens)
        if self.rpm_bool and self.add_rpm_after_cross_attention:
            video_tokens = video_tokens + self._rpm_tokens(rpm_idx, video.shape[0])

        batch_size = video.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) + self.video_position_embeddings[:, :1, :]
        sequence = torch.cat([cls_tokens, video_tokens], dim=1)
        sequence = self.dropout(sequence)
        encoded = self.encoder(sequence).last_hidden_state
        encoded = self.layernorm(encoded)
        pooled = encoded.mean(dim=1).contiguous()
        return self.fc(pooled)
