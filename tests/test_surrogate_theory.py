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
from scripts.measure_surrogate_geometry import subsample_indices
from config import SurrogateConfig
from losses import batched_visual_contrastive_loss, visual_contrastive_loss
from surrogates import Dinov2Wrapper


class _DinoOutput:
    def __init__(self, values):
        self.pooler_output = values


class _ToyDino(torch.nn.Module):
    def forward(self, pixel_values):
        return _DinoOutput(pixel_values.mean(dim=(-2, -1)))


class SurrogateTheoryTests(unittest.TestCase):
    def test_batched_attack_loss_matches_independent_item_mean(self):
        torch.manual_seed(7)
        batch, augmentations, positives, negatives, dimensions = 3, 2, 5, 4, 8
        images = torch.nn.functional.normalize(torch.randn(augmentations, batch, dimensions), dim=-1)
        positive = torch.nn.functional.normalize(torch.randn(batch, positives, dimensions), dim=-1)
        negative = torch.nn.functional.normalize(torch.randn(batch, negatives, dimensions), dim=-1)
        batched, _ = batched_visual_contrastive_loss(images, positive, negative, .1, 3, False)
        independent = []
        for augmentation in range(augmentations):
            for item in range(batch):
                value, _ = visual_contrastive_loss(
                    images[augmentation, item:item + 1], positive[item], negative[item], .1, 3, False,
                )
                independent.append(value)
        self.assertTrue(torch.allclose(batched, torch.stack(independent).mean(), atol=1e-6))

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

    def test_alignment_subsamples_are_deterministic_and_valid(self):
        first = subsample_indices(20, [5, 10, 30], 3, 7)
        second = subsample_indices(20, [5, 10, 30], 3, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual({size for size, _, _ in first}, {5, 10})
        self.assertTrue(all(len(indices) == size and len(set(indices)) == size for size, _, indices in first))

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

    def test_research_cycle_has_controlled_sets_and_exact_budget(self):
        experiment = yaml.safe_load(Path("configs/research_cycle_experiments.yaml").read_text())
        composition = load_composition_spec(experiment["composition_config"])
        self.assertEqual(
            composition.sets["four_vitb_redundant"].models,
            ("vit_b32_laion", "vit_b16_laion", "vit_b32_openai", "vit_b16_openai"),
        )
        self.assertEqual(
            composition.sets["four_architecture_mixed"].models,
            ("vit_b32_laion", "vit_b16_laion", "vit_b32_openai", "rn50_openai"),
        )
        trials = stage_trials(experiment, composition, "composition_screen")
        self.assertEqual(len(trials), 21)
        self.assertEqual({trial["forward_units_per_item"] for trial in trials}, {960})

    def test_cross_family_registry_is_explicit_and_disjoint(self):
        composition = load_composition_spec("configs/research_cycle_cross_family.yaml")
        families = {value.architecture_family for value in composition.models.values()}
        self.assertTrue({
            "openclip_vit", "openclip_resnet", "openclip_convnext",
            "siglip_vit", "eva02_vit", "dinov2_vit",
        }.issubset(families))
        self.assertEqual(composition.models["dinov2_base_control"].evaluation_role, "control")
        self.assertEqual(composition.models["dinov2_base_control"].backend, "huggingface_dinov2")
        for value in composition.sets.values():
            self.assertFalse(set(value.models).intersection(composition.heldout_models))

        experiment = yaml.safe_load(Path("configs/research_cycle_cross_family_experiments.yaml").read_text())
        trials = stage_trials(experiment, composition, "cross_family_screen")
        self.assertEqual(len(trials), 30)
        self.assertEqual({trial["forward_units_per_item"] for trial in trials}, {1200})

    def test_dinov2_wrapper_is_differentiable_and_normalized(self):
        config = SurrogateConfig(
            model_name="dinov2-toy",
            pretrained="local-toy",
            backend="huggingface_dinov2",
            use_fp16=False,
            patch_size=14,
        )
        wrapper = Dinov2Wrapper(config, _ToyDino(), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 14)
        images = torch.rand(2, 3, 28, 28, requires_grad=True)
        embeddings = wrapper.encode_image(images)
        self.assertEqual(tuple(embeddings.shape), (2, 3))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(embeddings, dim=-1), torch.ones(2)))
        embeddings.sum().backward()
        self.assertIsNotNone(images.grad)

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
