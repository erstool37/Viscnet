import argparse
import json
import tempfile
import unittest
from pathlib import Path

import scripts.build_real_window_dataset as builder


class BuildRealWindowDatasetTests(unittest.TestCase):
    def test_existing_manifest_still_ensures_backgrounds(self):
        old_root = builder.ROOT
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        builder.ROOT = root
        self.addCleanup(setattr, builder, "ROOT", old_root)

        source_root = root / "dataset" / "train"
        video_path = source_root / "videos" / "sample.mp4"
        backgrounds = source_root / "backgrounds"
        output_root = root / "outputs" / "windows"
        video_path.parent.mkdir(parents=True)
        backgrounds.mkdir(parents=True)
        output_root.mkdir(parents=True)
        video_path.write_bytes(b"placeholder")
        (backgrounds / "1.png").write_bytes(b"placeholder")

        source_manifest = root / "source_manifest.json"
        source_manifest.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "video_path": "dataset/train/videos/sample.mp4",
                            "parameters_norm_path": "dataset/train/parametersNorm/sample.json",
                        }
                    ]
                }
            )
        )
        (output_root / "manifest.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "video_path": "dataset/train/videos/sample.mp4",
                            "parameters_norm_path": "dataset/train/parametersNorm/sample.json",
                        }
                    ]
                }
            )
        )

        builder.build(
            argparse.Namespace(
                source_manifest="source_manifest.json",
                output_root="outputs/windows",
                window_size=30,
                windows_per_video=1,
                sample_stride=1,
                phase_offsets="0",
                max_records=None,
                force=False,
                allow_partial_phase_offsets=False,
                seed=37,
            )
        )

        self.assertTrue((output_root / "backgrounds").exists() or (output_root / "backgrounds").is_symlink())


if __name__ == "__main__":
    unittest.main()
