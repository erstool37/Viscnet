import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datasets.VideoDataset import VideoDataset  # noqa: E402
from datasets.augmentations import (  # noqa: E402
    apply_video_augmentation_consistently,
    build_video_augmentation,
)
from datasets.temporal import select_temporal_window  # noqa: E402


class FakeReplayTransform:
    def __call__(self, image):
        return {"image": image + 10, "replay": {"offset": 20}}


class VideoAugmentationTests(unittest.TestCase):
    def test_apply_video_augmentation_reuses_one_replay_for_every_remaining_frame(self):
        frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in (1, 2, 3)]

        with mock.patch(
            "datasets.augmentations.A.ReplayCompose.replay",
            side_effect=lambda replay, image: {"image": image + replay["offset"]},
        ) as replay:
            augmented = apply_video_augmentation_consistently(frames, FakeReplayTransform())

        self.assertEqual(augmented[0][0, 0, 0], 11)
        self.assertEqual(augmented[1][0, 0, 0], 22)
        self.assertEqual(augmented[2][0, 0, 0], 23)
        self.assertEqual(replay.call_count, 2)
        for call in replay.call_args_list:
            self.assertEqual(call.args[0], {"offset": 20})

    def test_build_video_augmentation_records_configured_policy_metadata(self):
        transform = build_video_augmentation(
            {"type": "augv2", "probability": 0.65, "strength": 1.2},
            output_size=224,
        )

        self.assertEqual(transform.policy_name, "augv2")
        self.assertEqual(transform.probability, 0.65)
        self.assertEqual(transform.strength, 1.2)

    def test_augv2_noise_includes_gaussian_sensor_noise(self):
        transform = build_video_augmentation(
            {"type": "augv2_noise", "probability": 0.9, "strength": 1.25},
            output_size=224,
        )

        transform_names = [type(item).__name__ for item in transform.transforms]
        self.assertEqual(transform.policy_name, "augv2_noise")
        self.assertIn("GaussNoise", transform_names)

    def test_video_dataset_accepts_config_driven_augmentation(self):
        dataset = VideoDataset(
            [],
            [],
            frame_num=10,
            time=5,
            aug_bool=True,
            visc_class=10,
            augmentation_config={"type": "augv1", "probability": 0.8, "strength": 1.0},
        )

        self.assertEqual(dataset.augmentation_config["type"], "augv1")
        self.assertIsNotNone(dataset.augmentation)

    def test_select_temporal_window_random_mode_returns_contiguous_clip(self):
        frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(50)]

        with mock.patch("datasets.temporal.random.randint", return_value=11):
            selected = select_temporal_window(frames, window_size=30, mode="random")

        self.assertEqual(len(selected), 30)
        self.assertEqual(selected[0][0, 0, 0], 11)
        self.assertEqual(selected[-1][0, 0, 0], 40)

    def test_select_temporal_window_first_mode_is_deterministic(self):
        frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(50)]

        selected = select_temporal_window(frames, window_size=30, mode="first")

        self.assertEqual(len(selected), 30)
        self.assertEqual(selected[0][0, 0, 0], 0)
        self.assertEqual(selected[-1][0, 0, 0], 29)


if __name__ == "__main__":
    unittest.main()
