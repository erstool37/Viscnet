import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import scripts.summarize_no_rpm_policy_results as summary  # noqa: E402


class NoRpmPolicySummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.original_output_root = summary.OUTPUT_ROOT
        summary.OUTPUT_ROOT = Path(self.tmpdir.name)
        self.addCleanup(self.restore_output_root)

    def restore_output_root(self):
        summary.OUTPUT_ROOT = self.original_output_root

    def write_metrics(self, run_name, accuracy, ece=None):
        metrics_dir = summary.OUTPUT_ROOT / run_name / "confusion_matrix"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / f"{run_name}_metrics.json").write_text(json.dumps({"accuracy": accuracy}))
        if ece is not None:
            reliability_dir = summary.OUTPUT_ROOT / run_name / "reliability_plots"
            reliability_dir.mkdir(parents=True)
            (reliability_dir / f"{run_name}_metrics.json").write_text(json.dumps({"ece": ece}))

    def write_log(self, run_name, text):
        log_dir = summary.OUTPUT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{run_name}.log").write_text(text)

    def test_run_record_marks_transfer_target_met_from_metrics(self):
        run_name = "repro_transfer_993_batch8_normal_no_rpm_from_synth_no_rpm_lr3e5_ep70"
        self.write_metrics(run_name, 0.9123, ece=0.041)
        self.write_log(
            run_name,
            "wandb: View run at https://wandb.ai/entity/re-rebuild-viscnet/runs/abc123xyz\n",
        )

        record = summary.run_record(
            {
                "name": run_name,
                "role": "synthetic-transfer target",
                "target": 0.9001,
            }
        )

        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["accuracy"], 0.9123)
        self.assertEqual(record["ece"], 0.041)
        self.assertEqual(record["wandb_run_id"], "abc123xyz")
        self.assertTrue(record["target_met"])

    def test_wandb_from_log_falls_back_to_local_run_id(self):
        run_name = "repro_synthetic_pretrain_sph35000_no_rpm_ep50"
        self.write_log(run_name, "wandb: Run data is saved locally in /root/Viscnet/wandb/run-20260526_063223-2wswho3w\n")

        wandb = summary.wandb_from_log(summary.log_path(run_name))

        self.assertEqual(wandb["run_id"], "2wswho3w")
        self.assertIsNone(wandb["url"])

    def test_markdown_reports_transfer_minus_best_realonly(self):
        records = [
            {
                "name": "real_a",
                "role": "real-only baseline",
                "status": "complete",
                "accuracy": 0.88,
                "ece": None,
                "target": None,
                "target_met": None,
                "wandb_run_id": "realid",
            },
            {
                "name": "real_b",
                "role": "real-only recovery",
                "status": "complete",
                "accuracy": 0.891,
                "ece": None,
                "target": None,
                "target_met": None,
                "wandb_run_id": None,
            },
            {
                "name": "transfer",
                "role": "synthetic-transfer target",
                "status": "complete",
                "accuracy": 0.912,
                "ece": None,
                "target": 0.9001,
                "target_met": True,
                "wandb_run_id": "transferid",
            },
        ]

        report = summary.markdown(records)

        self.assertIn("Transfer target met: accuracy 0.9120 >= 0.9001.", report)
        self.assertIn("Best no-RPM real-only result: `real_b` at 0.8910.", report)
        self.assertIn("Transfer minus best real-only: +0.0210.", report)
        self.assertIn("`transfer`: `transferid`", report)

    def test_acceptance_summary_requires_all_metrics_and_transfer_target(self):
        records = [
            {
                "name": "real",
                "role": "real-only baseline",
                "status": "complete",
                "accuracy": 0.88,
                "target_met": None,
            },
            {
                "name": "transfer",
                "role": "synthetic-transfer target",
                "status": "complete",
                "accuracy": 0.901,
                "target_met": True,
            },
            {
                "name": "dual",
                "role": "dual-pattern synthetic-transfer",
                "status": "missing_metrics",
                "accuracy": None,
                "target_met": None,
            },
        ]

        acceptance = summary.acceptance_summary(records)

        self.assertFalse(acceptance["accepted"])
        self.assertFalse(acceptance["all_metrics_present"])
        self.assertTrue(acceptance["transfer_target_met"])

    def test_acceptance_summary_accepts_complete_target_met_suite(self):
        records = [
            {
                "name": "real",
                "role": "real-only baseline",
                "status": "complete",
                "accuracy": 0.88,
                "target_met": None,
            },
            {
                "name": "transfer",
                "role": "synthetic-transfer target",
                "status": "complete",
                "accuracy": 0.901,
                "target_met": True,
            },
        ]

        acceptance = summary.acceptance_summary(records)

        self.assertTrue(acceptance["accepted"])
        self.assertTrue(acceptance["all_metrics_present"])
        self.assertTrue(acceptance["transfer_target_met"])


if __name__ == "__main__":
    unittest.main()
