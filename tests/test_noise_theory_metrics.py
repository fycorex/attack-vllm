import unittest

import torch

from noise_theory_metrics import (
    bootstrap_smoothing_diagnostics,
    clipping_statistics,
    jensen_gap,
    kernel_residual_statistics,
    smoothed_embeddings,
    smoothing_diagnostics,
)


class NoiseTheoryMetricsTests(unittest.TestCase):
    def test_smoothed_embeddings_distinguish_raw_and_renormalized_mean(self):
        samples = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
        raw, normalized = smoothed_embeddings(samples)
        self.assertLess(float(torch.linalg.vector_norm(raw[0])), 1.0)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(normalized[0])), 1.0, places=6)

    def test_jensen_gap_is_nonnegative(self):
        samples = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        gaps = jensen_gap(samples, anchors)
        self.assertTrue(bool((gaps >= -1e-6).all()))

    def test_identical_smoothing_has_zero_residual(self):
        proxy = torch.eye(3)
        result = kernel_residual_statistics(proxy, proxy, proxy, centered=False)
        self.assertAlmostEqual(result["residual_norm"], 0.0)
        self.assertAlmostEqual(result["alignment_delta"], 0.0)

    def test_diagnostics_keep_centered_and_uncentered_separate(self):
        proxy = torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
        samples = torch.stack([proxy, proxy.roll(1, 0)])
        output = smoothing_diagnostics(proxy, samples, proxy)
        self.assertEqual(
            set(output),
            {"raw_mean_uncentered", "raw_mean_centered", "renormalized_mean_uncentered", "renormalized_mean_centered"},
        )

    def test_clipping_statistics_use_effective_noise(self):
        original = torch.tensor([0.0, 0.5, 1.0])
        sampled = torch.stack([torch.clamp(original + 0.2, 0, 1), torch.clamp(original - 0.2, 0, 1)])
        result = clipping_statistics(original, sampled)
        self.assertGreater(result["saturated_fraction"], 0.0)
        self.assertLess(result["effective_noise_std"], 0.2)

    def test_bootstrap_diagnostics_are_seeded(self):
        proxy = torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
        samples = torch.stack([proxy, proxy.roll(1, 0)])
        first = bootstrap_smoothing_diagnostics(proxy, samples, proxy, bootstrap_samples=10, seed=7)
        second = bootstrap_smoothing_diagnostics(proxy, samples, proxy, bootstrap_samples=10, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["raw_mean_centered"]["bootstrap_samples"], 10)


if __name__ == "__main__":
    unittest.main()
