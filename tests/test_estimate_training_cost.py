import json
import tempfile
import unittest
from pathlib import Path

from scripts.estimate_training_cost import estimate_cost, load_profile_samples_per_sec


class EstimateTrainingCostTest(unittest.TestCase):
    def test_estimate_cost_uses_plan_formula(self):
        result = estimate_cost(
            samples_per_sec=1000.0,
            epochs=2,
            train_samples=3600,
            gpu_hour_price=3.0,
            num_gpus=2,
        )

        self.assertEqual(result["wall_hours"], 0.002)
        self.assertEqual(result["gpu_hours"], 0.004)
        self.assertEqual(result["cost_usd"], 0.012)

    def test_load_profile_samples_per_sec_averages_after_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_profile.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"samples_per_sec": 100.0}),
                        json.dumps({"samples_per_sec": 200.0}),
                        json.dumps({"samples_per_sec": 300.0}),
                    ]
                )
            )

            self.assertEqual(load_profile_samples_per_sec(path, skip_first=1), 250.0)

    def test_load_profile_samples_per_sec_uses_sample_weighted_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_profile.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "samples_per_sec": 1000.0,
                                "sample_count": 100.0,
                                "epoch_sec": 1.0,
                            }
                        ),
                        json.dumps(
                            {
                                "samples_per_sec": 10.0,
                                "sample_count": 900.0,
                                "epoch_sec": 9.0,
                            }
                        ),
                    ]
                )
            )

            self.assertEqual(load_profile_samples_per_sec(path), 100.0)


if __name__ == "__main__":
    unittest.main()
