import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.real_test_monitor import (  # noqa: E402
    compute_distribution_score,
    log_classification_real_test_monitor,
    log_regression_real_test_monitor,
    should_replace_diagnostic_checkpoint,
    should_run_real_test_monitor,
)


class FakeWandb:
    class Image:
        def __init__(self, path):
            self.path = path

    def __init__(self):
        self.records = []

    def log(self, payload):
        self.records.append(payload)


class RealTestMonitorTests(unittest.TestCase):
    def test_should_run_real_test_monitor_obeys_enabled_and_interval(self):
        self.assertFalse(should_run_real_test_monitor(1, {"enabled": False, "interval_epochs": 1}))
        self.assertTrue(should_run_real_test_monitor(1, {"enabled": True, "interval_epochs": 1}))
        self.assertFalse(should_run_real_test_monitor(2, {"enabled": True, "interval_epochs": 3}))
        self.assertTrue(should_run_real_test_monitor(3, {"enabled": True, "interval_epochs": 3}))

    def test_log_classification_real_test_monitor_writes_confusion_and_wandb_payload(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        output_root = Path(tmpdir.name)
        fake_wandb = FakeWandb()

        def fake_confusion(name, logits, labels, save_dir):
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / f"{name}.png").write_bytes(b"png")
            (save_path / f"{name}_metrics.json").write_text(
                json.dumps(
                    {
                        "accuracy": 0.5,
                        "labels": [0, 1],
                        "support": [1, 1],
                        "per_class_accuracy": [1.0, 0.0],
                        "confusion_matrix_counts": [[1, 0], [1, 0]],
                    }
                )
            )

        metrics = log_classification_real_test_monitor(
            run_name="example",
            epoch_number=7,
            mean_loss=1.25,
            logits=np.array([[2.0, 0.0], [1.0, 0.0]]),
            labels=np.array([0, 1]),
            output_root=output_root,
            wandb_module=fake_wandb,
            confusion_matrix_fn=fake_confusion,
        )

        self.assertEqual(metrics["accuracy"], 0.5)
        expected_dir = output_root / "real_test_monitor" / "epoch_007"
        self.assertTrue((expected_dir / "example_realtest_epoch007.png").exists())
        self.assertEqual(fake_wandb.records[0]["test_loss"], 1.25)
        self.assertEqual(fake_wandb.records[0]["epoch"], 7)
        self.assertEqual(set(fake_wandb.records[0]), {"epoch", "test_loss"})
        self.assertNotIn("real_test_zero_predicted_classes", fake_wandb.records[0])
        self.assertNotIn("real_test_confusion_matrix", fake_wandb.records[0])

    def test_compute_distribution_score_rewards_used_classes_and_penalizes_collapse(self):
        score = compute_distribution_score(
            {
                "predicted_classes_used": 8,
                "max_predicted_class_share": 0.30,
                "zero_predicted_classes": [1, 2],
            },
            class_count=10,
        )

        self.assertAlmostEqual(score, 0.30)

    def test_log_regression_real_test_monitor_logs_only_test_loss(self):
        fake_wandb = FakeWandb()

        metrics = log_regression_real_test_monitor(
            epoch_number=4,
            mean_loss=0.125,
            wandb_module=fake_wandb,
        )

        self.assertEqual(metrics, {"epoch": 4, "real_test_loss": 0.125})
        self.assertEqual(fake_wandb.records[0], {"epoch": 4, "test_loss": 0.125})

    def test_should_replace_diagnostic_checkpoint_prefers_distribution_over_accuracy(self):
        current = {
            "distribution_score": -0.20,
            "real_test_loss": 0.7,
            "accuracy": 0.50,
        }
        candidate = {
            "distribution_score": 0.10,
            "real_test_loss": 1.4,
            "accuracy": 0.10,
        }

        self.assertTrue(should_replace_diagnostic_checkpoint(current, candidate))


if __name__ == "__main__":
    unittest.main()
