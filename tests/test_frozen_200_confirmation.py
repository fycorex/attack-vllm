import tempfile
import unittest
from pathlib import Path

from scripts.run_frozen_200_attack_confirmation import (
    conditions,
    effective_config,
    matched_forward_budget,
)


class Frozen200ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.augmentation = {
            "mode": "gaussian_eot", "sigma": 2 / 255, "samples": 2,
            "geometry_mode": "translation", "geometry_samples": 2,
            "selection_score": 0.6,
        }
        self.rows = conditions(
            {"promoted": ["two_cross_objective"]}, {"promoted": [self.augmentation]}
        )

    def test_condition_matrix_keeps_baseline_and_combination(self):
        self.assertEqual(
            [row["condition"] for row in self.rows],
            ["baseline", "best_surrogate", "best_augmentation", "best_combined"],
        )

    def test_budget_is_exactly_matched_for_every_condition(self):
        budget = matched_forward_budget(self.rows)
        self.assertGreaterEqual(budget, 1200)
        self.assertEqual(budget, 2400)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in self.rows:
                config, record = effective_config(row, root / "manifest.json", root / row["condition"], 200, 42, budget)
                self.assertEqual(record["expected_forward_units_per_item"], budget)
                self.assertEqual(config["runtime"]["attack_limit"], 200)
                self.assertFalse(config["evaluation"]["gpt_victim"]["enabled"])


if __name__ == "__main__":
    unittest.main()
