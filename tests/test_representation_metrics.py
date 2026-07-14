import unittest

import torch

from representation_metrics import (
    centered_linear_cka,
    distance_kernel_audit,
    explained_alignment,
    gradient_stability,
    linear_gram,
    neighbor_margin,
    neighborhood_overlap,
    normalized_prototype,
    pearson_correlation,
    prototype_distance,
    uncentered_kernel_alignment,
)


class RepresentationMetricsTests(unittest.TestCase):
    def test_cka_identity_and_centered_invariance(self) -> None:
        values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]])
        self.assertAlmostEqual(centered_linear_cka(values, values), 1.0, places=6)
        self.assertAlmostEqual(uncentered_kernel_alignment(values, values), 1.0, places=6)
        transformed = values @ torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        self.assertAlmostEqual(centered_linear_cka(values, transformed), 1.0, places=6)

    def test_centered_and_uncentered_alignment_are_not_conflated(self) -> None:
        first = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        second = first + 10.0
        self.assertGreater(centered_linear_cka(first, second), 0.999)
        self.assertLess(uncentered_kernel_alignment(first, second), 0.95)

    def test_neighborhood_overlap_and_margin(self) -> None:
        first = torch.tensor([[0.0], [1.0], [3.0], [10.0]])
        self.assertEqual(neighborhood_overlap(first, first, 1), 1.0)
        margins = neighbor_margin(first, 1)
        self.assertEqual(margins.shape, (4,))
        self.assertTrue(torch.all(margins >= 0))

    def test_prototype_is_normalized_and_distance_is_zero_at_prototype(self) -> None:
        examples = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        prototype = normalized_prototype(examples)
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(prototype, dim=-1), torch.ones(1)))
        self.assertTrue(torch.allclose(prototype_distance(prototype, prototype), torch.zeros(1)))

    def test_explained_alignment_handles_redundant_kernels(self) -> None:
        values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        kernel = linear_gram(values)
        result = explained_alignment([kernel, kernel], kernel)
        self.assertEqual(result["gram_rank"], 1)
        self.assertAlmostEqual(result["explained_alignment"], 1.0, places=5)
        self.assertGreater(result["regularization"], 0)

    def test_exact_distance_kernel_inequality_has_no_violation(self) -> None:
        first = torch.nn.functional.normalize(torch.randn(8, 5), dim=-1)
        second = torch.nn.functional.normalize(torch.randn(8, 7), dim=-1)
        result = distance_kernel_audit(first, second)
        self.assertLess(result["maximum_pointwise_violation"], 1e-5)

    def test_correlation_and_gradient_stability(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(pearson_correlation(values, 2 * values + 3), 1.0, places=6)
        identical = torch.ones(3, 2, 2)
        result = gradient_stability(identical)
        self.assertEqual(result["gradient_component_variance"], 0.0)
        self.assertAlmostEqual(result["gradient_pairwise_cosine_mean"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
