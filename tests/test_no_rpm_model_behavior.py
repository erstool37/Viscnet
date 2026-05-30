import os
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.vivit.configuration_vivit import VivitConfig  # noqa: E402
from models.vivit.modeling_vivit import VivitEmbeddings  # noqa: E402
from models.VivitEmbed import VivitEmbed  # noqa: E402


class RaisingRpmEmbedding(nn.Module):
    def forward(self, rpm_idx):
        raise AssertionError("RPM embedding was accessed while rpm_bool is false")


class DummyPatternBackbone(nn.Module):
    def forward(self, pattern):
        return torch.ones(pattern.shape[0], 512)


class DummyFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, video, rpm_idx, pattern):
        batch_size = video.shape[0]
        return type("Output", (), {"last_hidden_state": torch.ones(batch_size, 2, self.config.hidden_size)})


class NoRpmModelBehaviorTests(unittest.TestCase):
    def test_vivit_embeddings_do_not_access_rpm_embedding_when_disabled(self):
        config = VivitConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            tubelet_size=(2, 16, 16),
            image_size=32,
            num_frames=2,
            num_channels=3,
            rpm_bool=False,
            pat_bool=False,
        )
        embeddings = VivitEmbeddings(config)
        embeddings.rpm_embed = RaisingRpmEmbedding()

        output = embeddings(
            torch.randn(2, 2, 3, 32, 32),
            torch.tensor([9, 3]),
            torch.zeros(2, 224, 224, 3),
        )

        self.assertEqual(tuple(output.shape), (2, 5, 32))

    def test_pattern_embedding_uses_configured_hidden_size_and_patch_count(self):
        config = VivitConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            tubelet_size=(2, 16, 16),
            image_size=32,
            num_frames=2,
            num_channels=3,
            rpm_bool=False,
            pat_bool=True,
            pat_mode="embedding",
        )
        with patch("models.vivit.modeling_vivit.timm.create_model", return_value=DummyPatternBackbone()):
            embeddings = VivitEmbeddings(config)

        output = embeddings(
            torch.randn(2, 2, 3, 32, 32),
            torch.tensor([9, 3]),
            torch.zeros(2, 224, 224, 3),
        )

        self.assertEqual(tuple(output.shape), (2, 5, 32))

    def test_pattern_backbone_stays_eval_during_training_forward(self):
        config = VivitConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            tubelet_size=(2, 16, 16),
            image_size=32,
            num_frames=2,
            num_channels=3,
            rpm_bool=False,
            pat_bool=True,
            pat_mode="embedding",
        )
        with patch("models.vivit.modeling_vivit.timm.create_model", return_value=DummyPatternBackbone()):
            embeddings = VivitEmbeddings(config)
        embeddings.train()

        embeddings(
            torch.randn(2, 2, 3, 32, 32),
            torch.tensor([9, 3]),
            torch.zeros(2, 224, 224, 3),
        )

        self.assertFalse(embeddings.pat_backbone.training)

    def test_embedding_only_pattern_mode_does_not_create_head_subtract_path(self):
        with patch("models.VivitEmbed.VivitModel", DummyFeatureExtractor):
            model = VivitEmbed(
                dropout=0.0,
                output_size=3,
                class_bool=True,
                visc_class=10,
                gmm_num=3,
                rpm_bool=False,
                pat_bool=True,
                num_frames=50,
                image_size=224,
                pat_mode="embedding",
            )

        self.assertEqual(model.pat_mode, "embedding")
        self.assertIsNone(model.pat_backbone)
        self.assertIsNone(model.pat_proj)

    def test_late_concat_pattern_mode_uses_head_pattern_without_early_token_injection(self):
        with (
            patch("models.VivitEmbed.VivitModel", DummyFeatureExtractor),
            patch("models.VivitEmbed.timm.create_model", return_value=DummyPatternBackbone()),
        ):
            model = VivitEmbed(
                dropout=0.0,
                output_size=3,
                class_bool=True,
                visc_class=10,
                gmm_num=3,
                rpm_bool=False,
                pat_bool=True,
                num_frames=50,
                image_size=224,
                pat_mode="late_concat",
                pattern_gate_init=0.01,
            )

        self.assertFalse(model.featureextractor.config.pat_bool)
        self.assertIsNotNone(model.pat_backbone)
        self.assertIsNotNone(model.pat_proj)
        self.assertAlmostEqual(float(model.pattern_gate.detach()), 0.01)
        self.assertEqual(model.fc[0].in_features, model.hidden_size * 2)

        logits = model(torch.randn(2, 50, 3, 224, 224), torch.tensor([0, 0]), torch.zeros(2, 224, 224, 3))

        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertFalse(model.pat_backbone.training)

    def test_late_residual_pattern_mode_uses_gated_residual_head_pattern(self):
        with (
            patch("models.VivitEmbed.VivitModel", DummyFeatureExtractor),
            patch("models.VivitEmbed.timm.create_model", return_value=DummyPatternBackbone()),
        ):
            model = VivitEmbed(
                dropout=0.0,
                output_size=3,
                class_bool=True,
                visc_class=10,
                gmm_num=3,
                rpm_bool=False,
                pat_bool=True,
                num_frames=50,
                image_size=224,
                pat_mode="late_residual",
                pattern_gate_init=0.01,
            )

        self.assertFalse(model.featureextractor.config.pat_bool)
        self.assertIsNotNone(model.pat_backbone)
        self.assertIsNotNone(model.pat_proj)
        self.assertAlmostEqual(float(model.pattern_gate.detach()), 0.01)
        self.assertEqual(model.fc[0].in_features, model.hidden_size)

        logits = model(torch.randn(2, 50, 3, 224, 224), torch.tensor([0, 0]), torch.zeros(2, 224, 224, 3))

        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertFalse(model.pat_backbone.training)


if __name__ == "__main__":
    unittest.main()
