import os
import sys
import tempfile
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.VivitPatternNormTransformer import (  # noqa: E402
    PatternVideoCrossAttention,
    VivitPatternNormTransformer,
)
from utils import load_weights  # noqa: E402


class RaisingRpmEmbedding(torch.nn.Module):
    def forward(self, rpm_idx):
        raise AssertionError("RPM embedding was accessed while rpm_bool is false")


class PatternNormTransformerTests(unittest.TestCase):
    def make_small_model(self, class_bool=True):
        return VivitPatternNormTransformer(
            dropout=0.0,
            output_size=3,
            class_bool=class_bool,
            visc_class=10,
            gmm_num=3,
            rpm_bool=False,
            pat_bool=True,
            num_frames=4,
            image_size=32,
            pattern_norm={
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "pattern_encoder_layers": 1,
                "video_encoder_layers": 1,
                "pattern_patch_size": 16,
                "num_cross_attention_heads": 4,
                "cross_attention_layers": 1,
            },
        )

    def test_cross_attention_preserves_video_token_shape(self):
        module = PatternVideoCrossAttention(dim=32, num_heads=4, dropout=0.0)
        video_tokens = torch.randn(2, 8, 32)
        pattern_tokens = torch.randn(2, 4, 32)

        output = module(video_tokens, pattern_tokens)

        self.assertEqual(tuple(output.shape), tuple(video_tokens.shape))

    def test_tokenizers_return_3d_spatial_tokens_without_global_pattern_pooling(self):
        model = self.make_small_model(class_bool=True)

        video_tokens = model.tokenize_video(torch.randn(2, 4, 3, 32, 32))
        pattern_tokens = model.tokenize_pattern(torch.randn(2, 32, 32, 3))
        normalized_tokens = model.pattern_video_cross_attention(video_tokens, pattern_tokens)

        self.assertEqual(video_tokens.ndim, 3)
        self.assertEqual(pattern_tokens.ndim, 3)
        self.assertEqual(tuple(pattern_tokens.shape), (2, 4, 32))
        self.assertEqual(tuple(normalized_tokens.shape), tuple(video_tokens.shape))

    def test_forward_accepts_video_rpm_and_pattern_and_returns_class_logits(self):
        model = self.make_small_model(class_bool=True)
        video = torch.randn(2, 4, 3, 32, 32)
        rpm_idx = torch.tensor([0, 1])
        pattern = torch.randn(2, 32, 32, 3)

        logits = model(video, rpm_idx, pattern)

        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_forward_does_not_access_rpm_embedding_when_rpm_disabled(self):
        model = self.make_small_model(class_bool=True)
        model.rpm_embed = RaisingRpmEmbedding()

        logits = model(
            torch.randn(2, 4, 3, 32, 32),
            torch.tensor([9, 7]),
            torch.randn(2, 32, 32, 3),
        )

        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_debug_mode_retains_last_cross_attention_weights(self):
        model = VivitPatternNormTransformer(
            dropout=0.0,
            output_size=3,
            class_bool=True,
            visc_class=10,
            gmm_num=3,
            rpm_bool=False,
            pat_bool=True,
            num_frames=4,
            image_size=32,
            pattern_norm={
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "pattern_encoder_layers": 1,
                "video_encoder_layers": 1,
                "pattern_patch_size": 16,
                "num_cross_attention_heads": 4,
                "cross_attention_layers": 1,
                "store_cross_attention": True,
            },
        )

        model(
            torch.randn(2, 4, 3, 32, 32),
            torch.tensor([0, 0]),
            torch.randn(2, 32, 32, 3),
        )
        attention = model.get_last_cross_attention()

        self.assertIsNotNone(attention)
        self.assertEqual(tuple(attention.shape), (2, 8, 4))

    def test_regression_forward_returns_three_outputs(self):
        model = self.make_small_model(class_bool=False)

        output = model(
            torch.randn(2, 4, 3, 32, 32),
            torch.tensor([0, 0]),
            torch.randn(2, 32, 32, 3),
        )

        self.assertEqual(tuple(output.shape), (2, 3))

    def test_load_weights_transfers_patternnorm_backbone_from_classification_to_regression(self):
        classification_model = self.make_small_model(class_bool=True)
        regression_model = self.make_small_model(class_bool=False)
        with torch.no_grad():
            classification_model.video_patch_embeddings.projection.weight.fill_(0.1234)
            regression_model.video_patch_embeddings.projection.weight.zero_()
            regression_model.fc[-1].weight.zero_()

        with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint:
            torch.save(classification_model.state_dict(), checkpoint.name)
            load_weights(regression_model, checkpoint.name)

        self.assertTrue(
            torch.allclose(
                regression_model.video_patch_embeddings.projection.weight,
                torch.full_like(regression_model.video_patch_embeddings.projection.weight, 0.1234),
            )
        )
        self.assertEqual(tuple(regression_model.fc[-1].weight.shape), (3, 192))
        self.assertTrue(torch.all(regression_model.fc[-1].weight == 0))


if __name__ == "__main__":
    unittest.main()
