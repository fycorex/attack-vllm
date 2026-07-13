import unittest

from scripts.analyze_geometric_experiments import aggregate_comparisons, compare_to_baseline


def trial(mode, geometry, seed, successes, margins):
    return {
        "stage": "equal", "dataset": "caption", "surrogate_group": "g", "seed": seed,
        "mode": mode, "sigma": 0.0, "samples": 1,
        "geometry_mode": geometry, "geometry_samples": 1,
        "condition_id": f"{mode}-{geometry}",
        "heldout": {"items": [
            {"model": "target", "item_id": f"item_{index}", "proxy_success": success, "margin_gain": margin}
            for index, (success, margin) in enumerate(zip(successes, margins))
        ]},
    }


class AnalyzeGeometricExperimentsTests(unittest.TestCase):
    def test_item_clustered_effect_across_seeds(self):
        trials = []
        for seed in (42, 123):
            trials.extend([
                trial("none", "none", seed, [False, False], [0.0, 0.0]),
                trial("none", "translation", seed, [True, False], [0.2, 0.0]),
            ])
        observations = compare_to_baseline(trials, samples=100)
        aggregate = aggregate_comparisons(observations, samples=100)[0]
        self.assertEqual(aggregate["paired_observations"], 4)
        self.assertEqual(aggregate["unique_item_clusters"], 2)
        self.assertEqual(aggregate["seeds"], 2)
        self.assertAlmostEqual(aggregate["asr_delta"], .5)
        self.assertAlmostEqual(aggregate["margin_gain_delta"], .1)


if __name__ == "__main__":
    unittest.main()
