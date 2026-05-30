import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from losses.CE import CE  # noqa: E402


class CrossEntropyLossTests(unittest.TestCase):
    def test_label_smoothing_is_configurable(self):
        loss = CE("unused", "unused", 0.05)

        self.assertEqual(loss.criterion.label_smoothing, 0.05)


if __name__ == "__main__":
    unittest.main()
