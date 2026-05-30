import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from losses.MSE import MSE  # noqa: E402


class MSELossTests(unittest.TestCase):
    def test_constructor_accepts_main_loss_signature(self):
        loss = MSE("unused-descaler", "unused-path", 0.0)

        self.assertIsInstance(loss, MSE)

    def test_uses_kinematic_viscosity_column(self):
        loss = MSE()
        pred = torch.tensor([[0.0, 10.0, 1.0]])
        target = torch.tensor([[0.0, 0.0, 3.0]])

        self.assertEqual(loss(pred, target).item(), 4.0)


if __name__ == "__main__":
    unittest.main()
