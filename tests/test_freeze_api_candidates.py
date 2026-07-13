import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.freeze_api_candidates import select_candidates
from transfer_eval import load_frozen_candidate_manifest


class FreezeApiCandidateTests(unittest.TestCase):
    def test_selects_by_heldout_macro_and_requires_complete_seeds(self):
        rows = []
        for name, asr in (("a", .4), ("b", .6)):
            for seed in (1, 2, 3):
                for target in ("t1", "t2"):
                    rows.append({"stage": "s", "budget_mode": "equal", "dataset": "d", "surrogate_set": name,
                        "seed": seed, "target": target, "asr": asr, "mean_margin_gain": asr / 10,
                        "items": 20, "valid_items": 20, "missing_items": 0, "directory": f"/tmp/{name}/{seed}"})
        rows.append({**rows[0], "surrogate_set": "incomplete", "asr": .99})
        selected = select_candidates(rows, stage="s", budget_mode="equal", top_k=1, minimum_seeds=3)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["surrogate_set"], "b")
        self.assertEqual(selected[0]["seeds"], [1, 2, 3])

    def test_rejects_candidate_with_inconsistent_target_coverage(self):
        rows = []
        for seed in (1, 2, 3):
            for target in (("t1", "t2") if seed != 3 else ("t1",)):
                rows.append({"stage": "s", "budget_mode": "equal", "dataset": "d", "surrogate_set": "broken",
                    "seed": seed, "target": target, "asr": .9, "mean_margin_gain": .1,
                    "items": 20, "valid_items": 20, "missing_items": 0, "directory": f"/tmp/{seed}"})
        selected = select_candidates(rows, stage="s", budget_mode="equal", top_k=1,
                                     minimum_seeds=3, minimum_targets=2)
        self.assertEqual(selected, [])

    def test_frozen_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); attack = root / "attack"; attack.mkdir()
            analysis = root / "analysis.csv"; analysis.write_text("x\n1\n")
            value = {"schema_version": 1, "selection_source": "heldout_open_source_only",
                "api_results_used_for_selection": False, "analysis_csv": str(analysis.resolve()),
                "analysis_csv_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
                "candidates": [{"dataset": "caption", "surrogate_set": "two", "rank_within_dataset": 1,
                                "attack_output_directories": [str(attack.resolve())]}]}
            value["freeze_hash"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            manifest = root / "frozen.json"; manifest.write_text(json.dumps(value))
            directories, info = load_frozen_candidate_manifest(manifest)
            self.assertEqual(directories, [attack.resolve()]); self.assertEqual(info["candidate_count"], 1)
            metadata = info["directory_metadata"][str(attack.resolve())]
            self.assertEqual(metadata["candidate_id"], "caption::two::1")
            value["candidates"][0]["attack_output_directories"] = []
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_frozen_candidate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
