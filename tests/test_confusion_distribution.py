import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.check_confusion_distribution import summarize_distribution  # noqa: E402


class ConfusionDistributionTests(unittest.TestCase):
    def test_summarize_distribution_reports_predicted_class_usage(self):
        metrics = {
            "accuracy": 0.42,
            "labels": list(range(10)),
            "support": [10] * 10,
            "per_class_accuracy": [0.1 * i for i in range(10)],
            "confusion_matrix_counts": [
                [2, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [2, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [2, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [2, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [2, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [2, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [2, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [2, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [2, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [2, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            ],
        }

        summary = summarize_distribution(metrics)

        self.assertEqual(summary["predicted_class_counts"]["0"], 20)
        self.assertEqual(summary["predicted_classes_used"], 9)
        self.assertEqual(summary["zero_predicted_classes"], [8])
        self.assertEqual(summary["max_predicted_class_share"], 0.2)
        self.assertTrue(summary["well_distributed"])
        self.assertEqual(summary["support"]["9"], 10)
        self.assertAlmostEqual(summary["per_class_accuracy"]["9"], 0.9)

    def test_summarize_distribution_flags_collapsed_predictions(self):
        metrics = {
            "labels": list(range(10)),
            "support": [10] * 10,
            "per_class_accuracy": [0.0] * 10,
            "confusion_matrix_counts": [[10, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(10)],
        }

        summary = summarize_distribution(metrics)

        self.assertEqual(summary["predicted_classes_used"], 1)
        self.assertEqual(summary["max_predicted_class_share"], 1.0)
        self.assertFalse(summary["well_distributed"])


if __name__ == "__main__":
    unittest.main()
