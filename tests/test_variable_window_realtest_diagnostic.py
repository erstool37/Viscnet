import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.run_variable_window_realtest_diagnostic import (  # noqa: E402
    default_window_starts,
    summarize_confusion_metrics,
)


class VariableWindowRealTestDiagnosticTests(unittest.TestCase):
    def test_default_window_starts_cover_all_30_frame_windows_from_first_50(self):
        self.assertEqual(default_window_starts(window_size=30, source_frame_count=50), list(range(21)))

    def test_summarize_confusion_metrics_reports_distribution(self):
        metrics = {
            "labels": list(range(4)),
            "confusion_matrix_counts": [
                [1, 0, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            "accuracy": 0.5,
        }

        summary = summarize_confusion_metrics(metrics)

        self.assertEqual(summary["predicted_class_counts"], {"0": 1, "1": 0, "2": 3, "3": 0})
        self.assertEqual(summary["predicted_classes_used"], 2)
        self.assertEqual(summary["zero_predicted_classes"], [1, 3])
        self.assertEqual(summary["max_predicted_class_share"], 0.75)


if __name__ == "__main__":
    unittest.main()
