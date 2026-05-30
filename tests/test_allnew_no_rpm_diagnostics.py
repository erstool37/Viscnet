import os
import sys
import unittest

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.run_allnew_no_rpm_diagnostics import group_summaries, parse_name_metadata, rpm_for_model  # noqa: E402


class AllnewNoRpmDiagnosticsTests(unittest.TestCase):
    def test_rpm_for_model_zeroes_rpm_when_disabled(self):
        rpm_idx = torch.tensor([[2], [7], [9]])

        result = rpm_for_model(rpm_idx, rpm_bool=False)

        self.assertEqual(result.tolist(), [0, 0, 0])

    def test_rpm_for_model_keeps_rpm_when_enabled(self):
        rpm_idx = torch.tensor([[2], [7], [9]])

        result = rpm_for_model(rpm_idx, rpm_bool=True)

        self.assertEqual(result.tolist(), [2, 7, 9])

    def test_group_summaries_reports_counts_accuracy_and_prediction_counts(self):
        records = [
            {
                "rpm_idx": 1,
                "prediction": 5,
                "correct": True,
                "temporal_entropy": 0.2,
                "spatial_entropy": 0.3,
                "temporal_peak_idx": 1.0,
                "temporal_peak_frac": 0.4,
                "early_mass": 0.2,
                "mid_mass": 0.3,
                "late_mass": 0.5,
                "center_distance": 0.1,
                "top10_spatial_mass": 0.2,
            },
            {
                "rpm_idx": 1,
                "prediction": 9,
                "correct": False,
                "temporal_entropy": 0.4,
                "spatial_entropy": 0.5,
                "temporal_peak_idx": 2.0,
                "temporal_peak_frac": 0.6,
                "early_mass": 0.1,
                "mid_mass": 0.2,
                "late_mass": 0.7,
                "center_distance": 0.2,
                "top10_spatial_mass": 0.3,
            },
            {
                "rpm_idx": 2,
                "prediction": 9,
                "correct": True,
                "temporal_entropy": 0.8,
                "spatial_entropy": 0.9,
                "temporal_peak_idx": 3.0,
                "temporal_peak_frac": 0.7,
                "early_mass": 0.4,
                "mid_mass": 0.4,
                "late_mass": 0.2,
                "center_distance": 0.3,
                "top10_spatial_mass": 0.4,
            },
        ]

        summaries = group_summaries(records, "rpm_idx")

        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["group_value"], 1)
        self.assertEqual(summaries[0]["count"], 2)
        self.assertEqual(summaries[0]["accuracy"], 0.5)
        self.assertEqual(summaries[0]["prediction_counts"], {"5": 1, "9": 1})
        self.assertAlmostEqual(summaries[0]["temporal_entropy_mean"], 0.3)

    def test_parse_name_metadata_reads_physical_rpm_and_viscosity(self):
        metadata = parse_name_metadata("decay_10fps_visc000.89274_rpm270_renderA")

        self.assertEqual(metadata["rpm_value"], 270)
        self.assertEqual(metadata["viscosity_value"], 0.89274)


if __name__ == "__main__":
    unittest.main()
