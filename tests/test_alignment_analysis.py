import unittest

from alignment_analysis import average_ranks, bootstrap_spearman, spearman


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


if __name__ == "__main__":
    unittest.main()
