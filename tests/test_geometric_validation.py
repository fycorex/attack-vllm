import math
from pathlib import Path
import tempfile
import unittest

import yaml

from PIL import Image

from scripts.validate_geometric_stage import finite_tree, image_linf
from scripts.run_noise_search import matrix


class GeometricValidationTests(unittest.TestCase):
    def test_finite_tree(self):
        self.assertTrue(finite_tree({"x": [1.0, {"y": 2}]}))
        self.assertFalse(finite_tree({"x": float("nan")}))
        self.assertFalse(finite_tree([math.inf]))

    def test_saved_image_linf(self):
        with tempfile.TemporaryDirectory() as td:
            clean, adversarial = Path(td) / "clean.png", Path(td) / "adversarial.png"
            Image.new("RGB", (2, 2), (100, 100, 100)).save(clean)
            Image.new("RGB", (2, 2), (102, 100, 100)).save(adversarial)
            self.assertAlmostEqual(image_linf(clean, adversarial), 2 / 255, places=6)

    def test_cross_dataset_stages_use_matched_nineteen_items(self):
        spec = yaml.safe_load(Path("configs/geometric_pilot.yaml").read_text())
        for stage in ("equal_steps_geometric", "equal_forward_geometric"):
            trials = matrix(spec, stage)
            self.assertTrue(trials)
            self.assertEqual({trial["items"] for trial in trials}, {19})

    def test_equal_forward_conditions_have_identical_budget(self):
        spec = yaml.safe_load(Path("configs/geometric_pilot.yaml").read_text())
        trials = matrix(spec, "equal_forward_geometric")
        units = set()
        for trial in trials:
            noise_samples = trial["samples"] if trial["mode"].endswith("_eot") else 1
            geometry_samples = trial["geometry_samples"] if trial["geometry_mode"] != "none" else 1
            units.add(trial["steps"] * noise_samples * geometry_samples)
        self.assertEqual(units, {400})

    def test_research_cycle_single_and_two_sample_conditions_match_budget(self):
        spec = yaml.safe_load(Path("configs/research_cycle_augmentation.yaml").read_text())
        trials = matrix(spec, "strict_equal_forward_screen")
        self.assertEqual(len(trials), 27)
        units = set()
        for trial in trials:
            noise_samples = trial["samples"] if trial["mode"].endswith("_eot") else 1
            geometry_samples = trial["geometry_samples"] if trial["geometry_mode"] != "none" else 1
            units.add(trial["steps"] * noise_samples * geometry_samples)
        self.assertEqual(units, {200})


if __name__ == "__main__":
    unittest.main()
