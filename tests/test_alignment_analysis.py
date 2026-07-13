import unittest

from alignment_analysis import average_ranks, bootstrap_spearman, clustered_bootstrap_spearman, spearman


class AlignmentAnalysisTests(unittest.TestCase):
    def test_average_ranks_handles_ties(self):
        self.assertEqual(average_ranks([3, 1, 1, 2]), [4.0, 1.5, 1.5, 3.0])

    def test_spearman_direction(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [30, 20, 10]), -1.0)

    def test_bootstrap_reports_small_sample_uncertainty(self):
        result = bootstrap_spearman([1, 2], [2, 3])
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["ci95"], [None, None])

    def test_clustered_bootstrap_resamples_unique_items(self):
        first = [1, 1, 2, 2, 3, 3]
        second = [10, 11, 20, 21, 30, 31]
        clusters = ["a", "a", "b", "b", "c", "c"]
        result = clustered_bootstrap_spearman(first, second, clusters, samples=100)
        self.assertEqual(result["pairs"], 6)
        self.assertEqual(result["clusters"], 3)
        self.assertAlmostEqual(result["spearman"], 0.9561828874675149)

    def test_clustered_bootstrap_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            clustered_bootstrap_spearman([1], [2], [])


if __name__ == "__main__":
    unittest.main()
