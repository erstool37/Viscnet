import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_PATH = Path(ROOT)
sys.path.insert(0, ROOT)

from scripts.verify_no_rpm_policy import verify_config  # noqa: E402


class NoRpmPolicyTests(unittest.TestCase):
    def write_config(self, rpm_bool, name="repro_test_no_rpm"):
        payload = f"""
name: {name}
model:
  embeddings:
    rpm_bool: {str(rpm_bool).lower()}
training:
  curr_bool: false
  curr_ckpt: repro_synthetic_pretrain_sph35000_no_rpm_ep50.pth
"""
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "config.yaml"
        path.write_text(payload)
        return path

    def write_transfer_config(self, curr_ckpt):
        payload = f"""
name: repro_transfer_test_no_rpm
model:
  embeddings:
    rpm_bool: false
training:
  curr_bool: true
  curr_ckpt: {curr_ckpt}
"""
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "config.yaml"
        path.write_text(payload)
        return path

    def test_verify_config_accepts_false_rpm_bool(self):
        result = verify_config(self.write_config(False))

        self.assertEqual(result["rpm_bool"], False)
        self.assertEqual(result["errors"], [])

    def test_verify_config_accepts_allnew_run_names(self):
        result = verify_config(self.write_config(False, name="allnew_synthetic_pretrain_sph35000_no_rpm_augv1_ep80"))

        self.assertEqual(result["errors"], [])

    def test_verify_config_accepts_cross_pattern_run_names(self):
        result = verify_config(self.write_config(False, name="crosspat_train345_test1_no_rpm_lateconcat_lr5e5_gate001_ep70"))

        self.assertEqual(result["errors"], [])

    def test_verify_config_rejects_true_rpm_bool(self):
        result = verify_config(self.write_config(True))

        self.assertEqual(result["rpm_bool"], True)
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("model.embeddings.rpm_bool must be false", result["errors"][0])

    def test_verify_config_rejects_rpm_trained_transfer_checkpoint(self):
        result = verify_config(self.write_transfer_config("repro_synthetic_pretrain_sph35000.pth"))

        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("curr_ckpt for no-RPM transfer must reference no_rpm weights", result["errors"][0])

    def test_allnew_synthetic_configs_enable_real_test_monitoring(self):
        for path in [
            ROOT_PATH / "configs/rebuild/retries/allnew_synthetic_pretrain_sph35000_no_rpm_augv1_ep80.yaml",
            ROOT_PATH / "configs/rebuild/retries/allnew_synthetic_pretrain_sph35000_no_rpm_augv2_ep80.yaml",
            ROOT_PATH
            / "configs/rebuild/retries/allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_ep70.yaml",
        ]:
            cfg = yaml.safe_load(path.read_text())
            monitor = cfg["training"]["real_test_monitor"]
            self.assertTrue(monitor["enabled"])
            self.assertEqual(monitor["interval_epochs"], 1)
            self.assertEqual(monitor["dataset"], "test")

    def test_window30_noise_config_is_no_rpm_random_temporal_augv2_noise(self):
        path = (
            ROOT_PATH
            / "configs/rebuild/retries/allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_ep70.yaml"
        )
        cfg = yaml.safe_load(path.read_text())

        self.assertFalse(cfg["model"]["embeddings"]["rpm_bool"])
        self.assertEqual(cfg["model"]["transformer"]["num_frames"], 30)
        self.assertEqual(cfg["dataset"]["train"]["dataloader"]["temporal_window"]["mode"], "random")
        self.assertEqual(cfg["dataset"]["train"]["dataloader"]["augmentation"]["type"], "augv2_noise")
        self.assertEqual(cfg["dataset"]["train"]["dataloader"]["batch_size"], 12)
        self.assertEqual(cfg["training"]["num_epochs"], 70)
        self.assertEqual(cfg["training"]["checkpoint_selection"]["metric"], "real_test_distribution_score")

    def test_realonly_window30x21_no_rpm_config_is_real_only(self):
        path = ROOT_PATH / "configs/rebuild/retries/realonly_993_window30x21_no_rpm_ep50.yaml"
        cfg = yaml.safe_load(path.read_text())

        self.assertEqual(cfg["project"], "re-rebuild-viscnet")
        self.assertEqual(cfg["name"], "repro_realonly_993_window30x21_no_rpm_ep50")
        self.assertFalse(cfg["model"]["embeddings"]["rpm_bool"])
        self.assertEqual(cfg["model"]["transformer"]["num_frames"], 30)
        self.assertEqual(cfg["dataset"]["train"]["dataloader"]["dataloader"], "VideoDatasetReal")
        self.assertIn("real_train_993_windows30_stride1", cfg["dataset"]["train"]["train_root"])
        self.assertEqual(cfg["dataset"]["train"]["dataloader"]["batch_size"], 16)
        self.assertEqual(cfg["training"]["num_epochs"], 50)
        self.assertEqual(cfg["training"]["checkpoint_selection"]["metric"], "real_test_distribution_score")
        self.assertIn("no synthetic", cfg["training"]["acceptance"]["note"])

    def test_realonly_50f_pattern_embed_config_is_no_rpm_fixed50(self):
        path = ROOT_PATH / "configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_embed_ep45.yaml"
        cfg = yaml.safe_load(path.read_text())

        self.assertEqual(cfg["name"], "repro_realonly_993_50f_no_rpm_pattern_embed_ep45")
        self.assertFalse(cfg["model"]["embeddings"]["rpm_bool"])
        self.assertTrue(cfg["model"]["embeddings"]["pat_bool"])
        self.assertEqual(cfg["model"]["embeddings"]["pat_mode"], "embedding")
        self.assertEqual(cfg["model"]["transformer"]["num_frames"], 50)
        self.assertEqual(cfg["dataset"]["train"]["dataloader"]["batch_size"], 8)
        self.assertEqual(cfg["training"]["num_epochs"], 45)
        self.assertNotIn("temporal_window", cfg["dataset"]["train"]["dataloader"])
        self.assertIn("No temporal window shifting", cfg["training"]["acceptance"]["note"])

    def test_realonly_50f_late_pattern_configs_are_no_rpm_fixed50(self):
        expected = {
            "lateconcat": "late_concat",
            "lateresidual": "late_residual",
        }
        for suffix, mode in expected.items():
            path = ROOT_PATH / f"configs/rebuild/retries/realonly_993_50f_no_rpm_pattern_{suffix}_ep45.yaml"
            cfg = yaml.safe_load(path.read_text())

            self.assertEqual(cfg["name"], f"repro_realonly_993_50f_no_rpm_pattern_{suffix}_ep45")
            self.assertFalse(cfg["model"]["embeddings"]["rpm_bool"])
            self.assertTrue(cfg["model"]["embeddings"]["pat_bool"])
            self.assertEqual(cfg["model"]["embeddings"]["pat_mode"], mode)
            self.assertEqual(cfg["model"]["embeddings"]["pattern_gate_init"], 0.01)
            self.assertEqual(cfg["model"]["transformer"]["num_frames"], 50)
            self.assertEqual(cfg["dataset"]["train"]["dataloader"]["batch_size"], 8)
            self.assertEqual(cfg["training"]["optimizer"]["lr"], 3.0e-05)
            self.assertEqual(cfg["training"]["num_epochs"], 45)
            self.assertNotIn("temporal_window", cfg["dataset"]["train"]["dataloader"])
            self.assertIn("No early token pattern injection", cfg["training"]["acceptance"]["note"])


if __name__ == "__main__":
    unittest.main()
