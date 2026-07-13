import math
from pathlib import Path
import unittest
import yaml

import torch

from augmentations import AttackAugmentationPipeline, effective_sigma, sample_noise
from config import AttackHyperParams


class NoiseTransferabilityTests(unittest.TestCase):
    def setUp(self):
        self.ref = torch.full((64, 3, 32, 32), 0.5)
        self.sigma = 0.08

    def test_none_identity(self):
        self.assertTrue(torch.equal(sample_noise("none", self.ref, self.sigma), torch.zeros_like(self.ref)))

    def test_gaussian_moments(self):
        noise = sample_noise("gaussian_eot", self.ref, self.sigma, generator=torch.Generator().manual_seed(1))
        self.assertAlmostEqual(noise.mean().item(), 0, delta=.002)
        self.assertAlmostEqual(noise.var().item(), self.sigma ** 2, delta=.00015)

    def test_uniform_variance_matched(self):
        noise = sample_noise("uniform_eot", self.ref, self.sigma, generator=torch.Generator().manual_seed(2))
        self.assertLessEqual(noise.abs().max().item(), math.sqrt(3) * self.sigma + 1e-6)
        self.assertAlmostEqual(noise.var().item(), self.sigma ** 2, delta=.00015)

    def test_rademacher_values_and_variance(self):
        noise = sample_noise("rademacher_eot", self.ref, self.sigma, generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.all((noise == noise.new_tensor(-self.sigma)) | (noise == noise.new_tensor(self.sigma))))
        self.assertAlmostEqual(noise.var().item(), self.sigma ** 2, delta=.00015)

    def test_reproducible_and_independent(self):
        a = sample_noise("gaussian_eot", self.ref, self.sigma, generator=torch.Generator().manual_seed(4))
        b = sample_noise("gaussian_eot", self.ref, self.sigma, generator=torch.Generator().manual_seed(4))
        c = sample_noise("gaussian_eot", self.ref, self.sigma, generator=torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(a, b)); self.assertFalse(torch.equal(a, c))

    def test_antithetic_pair_cancels(self):
        base = sample_noise("gaussian_eot", self.ref, self.sigma, generator=torch.Generator().manual_seed(6))
        self.assertTrue(torch.equal(base + (-base), torch.zeros_like(base)))

    def test_schedule(self):
        cfg = AttackHyperParams(noise_sigma=.1, noise_schedule="linear_decay")
        self.assertAlmostEqual(effective_sigma(cfg, .2, .5), .05)
        cfg.noise_schedule = "cosine_decay"
        self.assertAlmostEqual(effective_sigma(cfg, .2, .5), .05, places=6)

    def test_no_double_gaussian(self):
        cfg = AttackHyperParams(noise_mode="gaussian_eot", enable_gaussian=True, enable_crop=False, enable_pad=False, enable_jpeg=False)
        pipeline = AttackAugmentationPipeline(cfg, 32)
        explicit = torch.full_like(self.ref[:1], .01)
        out = pipeline(self.ref[:1], .1, noise=explicit)
        self.assertTrue(torch.allclose(out, self.ref[:1] + explicit))

    def test_clamping(self):
        cfg = AttackHyperParams(noise_mode="gaussian_eot", enable_crop=False, enable_pad=False, enable_jpeg=False)
        out = AttackAugmentationPipeline(cfg, 32)(self.ref[:1], .1, noise=torch.ones_like(self.ref[:1]))
        self.assertEqual(out.max().item(), 1.0)

    def test_validation(self):
        with self.assertRaises(ValueError): AttackHyperParams(noise_mode="bad").validate_noise()
        with self.assertRaises(ValueError): AttackHyperParams(noise_mode="antithetic_gaussian_eot", noise_samples=3).validate_noise()
        with self.assertRaises(ValueError): AttackHyperParams(geometry_mode="rotation").validate_noise()
        with self.assertRaises(ValueError): AttackHyperParams(geometry_samples=0).validate_noise()

    def test_explicit_geometry_is_seeded_and_shape_preserving(self):
        image = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
        for mode in ("translation", "resize_pad", "scale", "geometric_mixture"):
            cfg = AttackHyperParams(geometry_mode=mode, enable_jpeg=False, enable_gaussian=False)
            pipeline = AttackAugmentationPipeline(cfg, 32)
            a = pipeline(image, .1, generator=torch.Generator().manual_seed(17))
            b = pipeline(image, .1, generator=torch.Generator().manual_seed(17))
            self.assertEqual(a.shape, image.shape)
            self.assertTrue(torch.equal(a, b), mode)

    def test_geometry_none_is_identity_without_other_augmentations(self):
        cfg = AttackHyperParams(geometry_mode="none", enable_jpeg=False, enable_gaussian=False)
        image = self.ref[:1]
        output = AttackAugmentationPipeline(cfg, 32)(image, .1)
        self.assertTrue(torch.equal(output, image))

    def test_search_groups_are_disjoint(self):
        spec = yaml.safe_load(Path("configs/noise_search.yaml").read_text())
        for group in spec["surrogate_groups"].values():
            attack = {(x["model_name"], x["pretrained"]) for x in group["attack"]}
            heldout = {(x["model_name"], x["pretrained"]) for x in group["heldout"]}
            self.assertFalse(attack & heldout)


if __name__ == "__main__": unittest.main()
