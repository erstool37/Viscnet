import os
import unittest
from pathlib import Path


ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class NoRpmQueueScriptTests(unittest.TestCase):
    def read_script(self, name):
        return (ROOT / "scripts" / name).read_text()

    def test_transfer_policy_queue_runs_no_rpm_preflight_for_all_configs(self):
        script = self.read_script("run_no_rpm_transfer_policy_queue.sh")

        self.assertIn("python3 scripts/verify_no_rpm_policy.py", script)
        self.assertIn('"${SYNTH_CONFIG}"', script)
        self.assertIn('"${TRANSFER_CONFIG}"', script)
        self.assertIn('"${DUAL_CONFIG}"', script)

    def test_realonly_followup_queue_runs_no_rpm_preflight(self):
        script = self.read_script("run_no_rpm_realonly_followup_queue.sh")

        self.assertIn("python3 scripts/verify_no_rpm_policy.py", script)
        self.assertIn('"${CONFIG}"', script)


if __name__ == "__main__":
    unittest.main()
