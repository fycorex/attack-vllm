import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_noise_theory import aggregate, read_run_list
from scripts.run_noise_theory_gated_analysis import REQUIRED, completed_runs


class NoiseTheoryAnalysisTests(unittest.TestCase):
    def _write(self, path: Path, rows: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    def test_aggregation_and_pairing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, mode, asr, target_asr, relevance, success in (("base", "none", .2, .1, -.1, [False, False]),
                                                                    ("noise", "gaussian_eot", .4, .5, .2, [True, False])):
                run = root / name; run.mkdir(); (run / "run_manifest.json").write_text("{}")
                attack = root / f"{name}_trial" / "attack"; attack.mkdir(parents=True)
                (attack.parent / "heldout_summary.json").write_text(json.dumps({
                    "heldout_models": {"t": {"asr": target_asr, "mean_margin_gain": .3}},
                    "items": [{"item_id": f"i{index}", "model": "t", "proxy_success": value}
                              for index, value in enumerate(success)]}))
                self._write(run / "residual_assumption_tests.csv", [{"kernel_definition": "raw_mean_centered",
                    "residual_orthogonality": .05, "residual_target_relevance": relevance,
                    "alignment_delta": .1, "discrepancy_delta": -.2}])
                self._write(run / "alignment_asr_join.csv", [{"dataset": "d", "surrogate_group": "s", "seed": 1,
                    "proxy": "p", "target": "t", "mode": mode, "heldout_macro_asr": asr,
                    "attack_output": str(attack),
                    "raw_centered_alignment_delta": .1, "raw_centered_discrepancy_delta": -.2,
                    "raw_centered_residual_orthogonality": .05, "raw_centered_residual_target_relevance": relevance}])
            result = aggregate(root, bootstrap_samples=20)
            self.assertEqual(len(result["paired"]), 1)
            self.assertAlmostEqual(result["paired"][0]["delta_heldout_asr"], .4)
            self.assertEqual(result["paired"][0]["paired_items"], 2)
            self.assertTrue(result["paired"][0]["used_target_specific_heldout"])
            self.assertEqual(result["overall_paired_effect"]["unique_item_clusters"], 2)
            summary = result["assumption_summary"][0]
            self.assertEqual(summary["alignment_improvement_evidence"], "supported")
            self.assertEqual(summary["residual_target_irrelevance_evidence"], "partially supported")

    def test_gate_requires_every_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); run = root / "run"; run.mkdir()
            for name in REQUIRED[:-1]:
                (run / name).write_text("x")
            self.assertEqual(completed_runs(root), [])
            (run / REQUIRED[-1]).write_text("x")
            self.assertEqual(completed_runs(root), [run])

    def test_exact_run_list_excludes_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); wanted = root / "wanted"; smoke = root / "smoke"
            wanted.mkdir(); smoke.mkdir()
            for directory in (wanted, smoke):
                for name in REQUIRED:
                    (directory / name).write_text("x")
            run_list = root / "runs.txt"; run_list.write_text("# matrix\nwanted\n")
            names = read_run_list(run_list)
            self.assertEqual(names, ["wanted"])
            self.assertEqual(completed_runs(root, names), [wanted])


if __name__ == "__main__":
    unittest.main()
