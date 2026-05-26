import os
import sys
import unittest

import torch
from torch import nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.vivit.configuration_vivit import VivitConfig  # noqa: E402
from models.vivit.modeling_vivit import VivitEmbeddings  # noqa: E402


class RaisingRpmEmbedding(nn.Module):
    def forward(self, rpm_idx):
        raise AssertionError("RPM embedding was accessed while rpm_bool is false")


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


if __name__ == "__main__":
    unittest.main()
