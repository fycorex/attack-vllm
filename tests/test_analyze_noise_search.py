import random
import unittest

from scripts.analyze_noise_search import clustered_bootstrap


class AnalyzeNoiseSearchTests(unittest.TestCase):
    def test_shared_source_questions_form_one_cluster(self):
        rows = [
            {"cluster_id": "image_a", "success_delta": 1.0},
            {"cluster_id": "image_a", "success_delta": 0.0},
            {"cluster_id": "image_b", "success_delta": 0.0},
        ]
        estimate, interval, clusters = clustered_bootstrap(rows, "success_delta", 100, random.Random(0))
        self.assertEqual(clusters, 2)
        self.assertAlmostEqual(estimate, .25)
        self.assertEqual(len(interval), 2)


if __name__ == "__main__":
    unittest.main()
