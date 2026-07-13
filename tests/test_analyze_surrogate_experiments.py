import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_surrogate_experiments.py"
SPEC = importlib.util.spec_from_file_location("analyze_surrogate_experiments", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _trial(method: str, seed: int, successes: list[bool]) -> dict:
    return {
        "stage": "equal_forwards",
        "dataset": "caption",
        "seed": seed,
        "surrogate_set": method,
        "item_success": {
            ("heldout_a", f"item_{index}"): success
            for index, success in enumerate(successes)
        },
    }


class AnalyzeSurrogateExperimentsTests(unittest.TestCase):
    def test_all_method_pairs_and_clustered_item_bootstrap(self):
        trials = []
        for seed in (42, 123):
            trials.extend([
                _trial("single_reference", seed, [False, False]),
                _trial("two_homogeneous", seed, [True, False]),
                _trial("four_lightweight_mixed", seed, [True, True]),
            ])

        cells, aggregates = MODULE.paired_method_comparisons(trials, samples=200)

        # Three unordered method pairs for each of two seeds and one target.
        self.assertEqual(len(cells), 6)
        pair_names = {(row["baseline"], row["candidate"]) for row in cells}
        self.assertIn(("two_homogeneous", "four_lightweight_mixed"), pair_names)

        macro = next(
            row for row in aggregates
            if row["scope"] == "macro_targets"
            and row["baseline"] == "single_reference"
            and row["candidate"] == "two_homogeneous"
        )
        self.assertEqual(macro["unique_item_clusters"], 2)
        self.assertEqual(macro["paired_observations"], 4)
        self.assertEqual(macro["seed_count"], 2)
        self.assertAlmostEqual(macro["asr_delta"], 0.5)

    def test_only_common_items_are_paired(self):
        baseline = _trial("single_reference", 42, [False, False])
        candidate = _trial("two_homogeneous", 42, [True])

        cells, _ = MODULE.paired_method_comparisons([baseline, candidate], samples=20)

        self.assertEqual(cells[0]["baseline_items"], 2)
        self.assertEqual(cells[0]["candidate_items"], 1)
        self.assertEqual(cells[0]["paired_items"], 1)
        self.assertEqual(cells[0]["asr_delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
