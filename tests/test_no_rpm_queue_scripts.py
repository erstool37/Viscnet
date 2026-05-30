import os
import unittest
from pathlib import Path


ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class NoRpmQueueScriptTests(unittest.TestCase):
    def read_script(self, name):
        return (ROOT / "scripts" / name).read_text()

    def test_allnew_queue_sources_env_requires_wandb_and_runs_aug_preflight(self):
        script = self.read_script("run_allnew_no_rpm_aug_queue.sh")

        self.assertIn(". ./.env", script)
        self.assertIn("WANDB_API_KEY is required", script)
        self.assertIn('export WANDB_PROJECT="allnewViscnet"', script)
        self.assertIn('export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', script)
        self.assertIn("python3 scripts/verify_no_rpm_policy.py", script)
        self.assertIn("allnew_synthetic_pretrain_sph35000_no_rpm_augv1_ep80.yaml", script)
        self.assertIn("allnew_synthetic_pretrain_sph35000_no_rpm_augv2_ep80.yaml", script)
        self.assertIn("allnew_synth_no_rpm_augv1_realtest_frozen_eval.yaml", script)
        self.assertIn("allnew_synth_no_rpm_augv2_realtest_frozen_eval.yaml", script)
        self.assertIn("skip_if_checkpoint_exists", script)
        self.assertIn("scripts/check_confusion_distribution.py", script)

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

    def test_window30_noise_queue_runs_no_rpm_preflight_and_allnew_project(self):
        script = self.read_script("run_allnew_no_rpm_augv2noise_window30_queue.sh")

        self.assertIn(". ./.env", script)
        self.assertIn("WANDB_API_KEY is required", script)
        self.assertIn('export WANDB_PROJECT="allnewViscnet"', script)
        self.assertIn('export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', script)
        self.assertIn("python3 scripts/verify_no_rpm_policy.py", script)
        self.assertIn("allnew_synthetic_pretrain_sph35000_no_rpm_augv2noise_window30_randtemporal_diag_ep70.yaml", script)
        self.assertIn("scripts/check_confusion_distribution.py", script)

    def test_realonly_window30x21_no_rpm_queue_has_no_synthetic_resume(self):
        script = self.read_script("run_realonly_window30x21_no_rpm_train_then_varwin.sh")

        self.assertIn(". ./.env", script)
        self.assertIn("WANDB_API_KEY is required", script)
        self.assertIn('export WANDB_PROJECT="re-rebuild-viscnet"', script)
        self.assertIn('export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"', script)
        self.assertIn("realonly_993_window30x21_no_rpm_ep50.yaml", script)
        self.assertIn("python3 scripts/verify_no_rpm_policy.py", script)
        self.assertIn("scripts/build_real_window_dataset.py", script)
        self.assertIn("scripts/run_variable_window_realtest_diagnostic.py", script)
        self.assertIn('"wandb_metric_policy": ["train_loss", "val_loss", "test_loss"]', script)
        self.assertNotIn("allnew_synthetic", script)

    def test_realonly_50f_pattern_queue_is_fixed_no_rpm_pattern_run(self):
        script = self.read_script("run_realonly_50f_no_rpm_pattern_embed_queue.sh")

        self.assertIn(". ./.env", script)
        self.assertIn("WANDB_API_KEY is required", script)
        self.assertIn('export WANDB_PROJECT="re-rebuild-viscnet"', script)
        self.assertIn('export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', script)
        self.assertIn("realonly_993_50f_no_rpm_pattern_embed_ep45.yaml", script)
        self.assertIn("python3 scripts/verify_no_rpm_policy.py", script)
        self.assertIn("backgrounds/1.png", script)
        self.assertIn("no temporal window shifting", script)
        self.assertNotIn("run_variable_window_realtest_diagnostic.py", script)

    def test_realonly_50f_late_pattern_queue_runs_both_variants(self):
        script = self.read_script("run_realonly_50f_no_rpm_late_pattern_queue.sh")

        self.assertIn(". ./.env", script)
        self.assertIn("WANDB_API_KEY is required", script)
        self.assertIn('export WANDB_PROJECT="re-rebuild-viscnet"', script)
        self.assertIn('export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', script)
        self.assertIn("realonly_993_50f_no_rpm_pattern_lateconcat_ep45.yaml", script)
        self.assertIn("realonly_993_50f_no_rpm_pattern_lateresidual_ep45.yaml", script)
        self.assertIn("python3 scripts/verify_no_rpm_policy.py", script)
        self.assertIn("backgrounds/1.png", script)
        self.assertIn("late_concat", script)
        self.assertIn("late_residual", script)
        self.assertNotIn("run_variable_window_realtest_diagnostic.py", script)

    def test_realonly_50f_pattern_hparam_grid_queue_is_val_loss_no_rpm(self):
        script = self.read_script("run_realonly_50f_no_rpm_pattern_hparam_grid_queue.sh")

        self.assertIn(". ./.env", script)
        self.assertIn("WANDB_API_KEY is required", script)
        self.assertIn('export WANDB_PROJECT="allnewviscnet"', script)
        self.assertIn('export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', script)
        self.assertIn("python3 scripts/verify_no_rpm_policy.py", script)
        self.assertIn('"checkpoint_selection": "val_loss"', script)
        self.assertIn('"threshold": 0.80', script)
        self.assertIn("late_concat", script)
        self.assertIn("late_residual", script)
        self.assertIn("lr3e5_gate001", script)
        self.assertIn("lr5e5_gate005", script)
        self.assertIn("pattern_gate_init", script)
        self.assertNotIn("real_test_distribution_score", script)
        self.assertNotIn("run_variable_window_realtest_diagnostic.py", script)

    def test_cross_pattern_1500_500_queue_is_no_rpm_val_loss(self):
        script = self.read_script("run_cross_pattern_1500_500_no_rpm_lateconcat_queue.sh")

        self.assertIn(". ./.env", script)
        self.assertIn("WANDB_API_KEY is required", script)
        self.assertIn('export WANDB_PROJECT="allnewviscnet"', script)
        self.assertIn('export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', script)
        self.assertIn("python3 scripts/verify_no_rpm_policy.py", script)
        self.assertIn("real_20rpm_increment_train345", script)
        self.assertIn("real_20rpm_increment_test1", script)
        self.assertIn("real_20rpm_increment_train134", script)
        self.assertIn("real_20rpm_increment_test5", script)
        self.assertIn("real_20rpm_increment_train135", script)
        self.assertIn("real_20rpm_increment_test4", script)
        self.assertIn('"checkpoint_selection": "val_loss"', script)
        self.assertIn('"pat_mode": "late_concat"', script)
        self.assertIn('"rpm_bool": False', script)
        self.assertNotIn("real_test_distribution_score", script)


if __name__ == "__main__":
    unittest.main()
