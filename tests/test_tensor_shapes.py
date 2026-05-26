import os
import sys
import unittest

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.tensor_shapes import as_batch_vector  # noqa: E402


class TensorShapeTests(unittest.TestCase):
    def test_as_batch_vector_keeps_single_item_batch_sliceable(self):
        rpm_idx = torch.tensor([3])

        normalized = as_batch_vector(rpm_idx, dtype=torch.long)

        self.assertEqual(tuple(normalized.shape), (1,))
        self.assertEqual(normalized[0:1].tolist(), [3])

    def test_as_batch_vector_flattens_collated_column_vector(self):
        rpm_idx = torch.tensor([[2], [4]])

        normalized = as_batch_vector(rpm_idx, dtype=torch.long)

        self.assertEqual(tuple(normalized.shape), (2,))
        self.assertEqual(normalized.tolist(), [2, 4])


if __name__ == "__main__":
    unittest.main()
