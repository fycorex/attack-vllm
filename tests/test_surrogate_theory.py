from pathlib import Path
import json
import tempfile
import unittest

import torch
import yaml
from PIL import Image

from surrogate_composition import build_attack_config, gradient_diagnostics, load_composition_spec, stage_trials
from experiment_validation import validate_attack_output
from theory_metrics import (
    centered_linear_cka,
    ensemble_theory_metrics,
    linear_gram,
    neighborhood_overlap,
    uncentered_kernel_alignment,
)


class SurrogateTheoryTests(unittest.TestCase):
    def test_config_resolves_exact_sets_and_metadata(self):
        spec = load_composition_spec("configs/surrogate_composition.yaml")
        resolved = spec.resolve("two_homogeneous")
        self.assertEqual([model.model_name for model in resolved], ["ViT-B-32", "ViT-B-16"])
        self.assertTrue(all(model.architecture_family and model.pretraining_family for model in spec.models.values()))

    def test_attack_heldout_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("""
models:
  x: {model_name: X, pretrained: p, input_size: 1, architecture_family: a, objective_family: o, pretraining_family: p}
heldout_models: [x]
sets: {bad: {models: [x], rationale: bad}}
""")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_composition_spec(path)

    def test_centered_and_uncentered_are_distinct(self):
        first = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        shifted = first + 10
        self.assertGreater(centered_linear_cka(first, shifted), .999)
        self.assertLess(uncentered_kernel_alignment(first, shifted), .95)

    def test_ensemble_metrics_redundant_and_orthogonal_limits(self):
        target = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]])
        duplicate = ensemble_theory_metrics([target, target], target, centered=False)
        self.assertEqual(duplicate["gram_rank"], 1)
        self.assertAlmostEqual(duplicate["explained_alignment"], 1.0, places=5)
        self.assertAlmostEqual(duplicate["mean_inter_proxy_alignment"], 1.0, places=6)
        self.assertAlmostEqual(neighborhood_overlap(target, target, 1), 1.0, places=6)

    def test_identity_gram(self):
        values = torch.eye(4)
        self.assertTrue(torch.equal(linear_gram(values), torch.eye(4)))

    def test_gradient_diagnostics_handles_zero_gradients(self):
        result = gradient_diagnostics({"a": torch.zeros(2), "b": torch.ones(2)})
        self.assertEqual(result["models"]["a"]["gradient_norm"], 0.0)
        self.assertEqual(result["pairwise_cosine"]["a__b"], 0.0)
        self.assertEqual(result["models"]["b"]["effective_equal_weight"], 0.5)

    def test_equal_forward_budget_and_api_disable(self):
        experiment = yaml.safe_load(Path("configs/surrogate_experiments.yaml").read_text())
        composition = load_composition_spec(experiment["composition_config"])
        trials = stage_trials(experiment, composition, "stage2_equal_forwards")
        self.assertEqual({trial["forward_units_per_item"] for trial in trials}, {1600})
        trial = next(trial for trial in trials if trial["surrogate_set"] == "two_homogeneous")
        base = yaml.safe_load(Path(experiment["datasets"][trial["dataset"]]["base_attack_config"]).read_text())
        generated = build_attack_config(base, experiment, composition, trial, Path("out"))
        self.assertEqual(len(generated["surrogates"]), 2)
        for value in generated["evaluation"].values():
            if isinstance(value, dict) and "enabled" in value:
                self.assertFalse(value["enabled"])

    def test_cross_dataset_equal_forward_budget(self):
        experiment = yaml.safe_load(Path("configs/surrogate_experiments.yaml").read_text())
        composition = load_composition_spec(experiment["composition_config"])
        trials = stage_trials(experiment, composition, "stage3_cross_dataset_equal_forwards")
        self.assertEqual({trial["forward_units_per_item"] for trial in trials}, {4800})
        steps = {trial["surrogate_set"]: trial["steps"] for trial in trials if trial["dataset"] == "caption_caltech" and trial["seed"] == 42}
        self.assertEqual(steps, {"single_reference": 1200, "two_homogeneous": 600, "four_lightweight_mixed": 300})

    def test_attack_output_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item_dir = root / "x"
            item_dir.mkdir()
            Image.new("RGB", (2, 2), color=(100, 100, 100)).save(item_dir / "clean.png")
            Image.new("RGB", (2, 2), color=(101, 100, 100)).save(item_dir / "adversarial.png")
            metrics = {"item_id": "x", "composition_diagnostics": {"total_surrogate_forwards": 80}, "value": 1.0}
            (item_dir / "metrics.json").write_text(json.dumps(metrics))
            (root / "summary.json").write_text(json.dumps({"items": [metrics]}))
            result = validate_attack_output(root, expected_items=1, epsilon=8/255, expected_forward_units_per_item=80)
            self.assertTrue(result["valid"])
            metrics["value"] = float("nan")
            (item_dir / "metrics.json").write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                validate_attack_output(root, expected_items=1, epsilon=8/255, expected_forward_units_per_item=80)


if __name__ == "__main__":
    unittest.main()
