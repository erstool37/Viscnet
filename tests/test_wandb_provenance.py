import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.provenance import build_wandb_config, resolve_wandb_project  # noqa: E402


class WandbProvenanceTests(unittest.TestCase):
    def test_env_project_allows_allnew_rebuild_runs(self):
        project = resolve_wandb_project(
            {"project": "viscnet-rebuild"},
            "configs/rebuild/retries/allnew.yaml",
            env={"WANDB_PROJECT": "allnewViscnet"},
        )

        self.assertEqual(project, "allnewViscnet")

    def test_legacy_rebuild_project_still_maps_to_rerebuild_without_override(self):
        project = resolve_wandb_project(
            {"project": "viscnet-rebuild"},
            "configs/rebuild/realonly_993.yaml",
            env={},
        )

        self.assertEqual(project, "re-rebuild-viscnet")

    def test_build_wandb_config_adds_launch_provenance_without_mutating_config(self):
        config = {"project": "allnewViscnet", "name": "example"}
        metadata = {
            "git_commit": "abc123",
            "git_branch": "allnew-no-rpm-aug-training",
            "git_dirty": True,
            "config_path": "configs/rebuild/retries/allnew.yaml",
            "launch_command": "torchrun src/main.py -c configs/rebuild/retries/allnew.yaml",
        }

        wandb_config = build_wandb_config(config, metadata)

        self.assertEqual(wandb_config["provenance"], metadata)
        self.assertNotIn("provenance", config)


if __name__ == "__main__":
    unittest.main()
