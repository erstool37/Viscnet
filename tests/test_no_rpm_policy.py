import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.verify_no_rpm_policy import verify_config  # noqa: E402


class NoRpmPolicyTests(unittest.TestCase):
    def write_config(self, rpm_bool):
        payload = f"""
name: repro_test_no_rpm
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

    def test_verify_config_rejects_true_rpm_bool(self):
        result = verify_config(self.write_config(True))

        self.assertEqual(result["rpm_bool"], True)
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("model.embeddings.rpm_bool must be false", result["errors"][0])

    def test_verify_config_rejects_rpm_trained_transfer_checkpoint(self):
        result = verify_config(self.write_transfer_config("repro_synthetic_pretrain_sph35000.pth"))

        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("curr_ckpt for no-RPM transfer must reference no_rpm weights", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
