import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.select_transfer_candidates import promote, score_candidates
from scripts.run_transfer_search_cycle import select_augmentations


class SelectTransferCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.config = Path(self.directory.name) / "composition.yaml"
        self.config.write_text(yaml.safe_dump({
            "models": {
                "a": {"model_name": "a", "pretrained": "a", "input_size": 1, "architecture_family": "p", "objective_family": "clip", "pretraining_family": "x"},
                "b": {"model_name": "b", "pretrained": "b", "input_size": 1, "architecture_family": "q", "objective_family": "siglip", "pretraining_family": "y"},
                "t1": {"model_name": "t1", "pretrained": "t1", "input_size": 1, "architecture_family": "f1", "objective_family": "clip", "pretraining_family": "z"},
                "t2": {"model_name": "t2", "pretrained": "t2", "input_size": 1, "architecture_family": "control", "objective_family": "visual", "pretraining_family": "z", "evaluation_role": "control"},
            },
            "heldout_models": ["t1", "t2"],
            "sets": {
                "single": {"models": ["a"]},
                "mixed": {"models": ["a", "b"]},
            },
        }))

    def tearDown(self):
        self.directory.cleanup()

    @staticmethod
    def rows():
        rows = []
        for candidate, primary, control in [("single", .5, .9), ("mixed", .8, .1)]:
            for dataset in ("caption", "vqa"):
                for target, asr in (("t1", primary), ("t2", control)):
                    rows.append({
                        "stage": "screen", "budget_mode": "equal_forwards",
                        "forward_units_per_item": "100", "surrogate_set": candidate,
                        "dataset": dataset, "seed": "42", "target": target,
                        "asr": str(asr), "missing_items": "0",
                    })
        return rows

    def test_control_does_not_drive_selection(self):
        ranking = score_candidates(self.rows(), self.config, "screen")
        self.assertEqual(ranking[0]["surrogate_set"], "mixed")
        self.assertEqual(ranking[0]["control_asr"], {"t2": .1})
        self.assertEqual(promote(ranking, 2), ["mixed", "single"])

    def test_rejects_api_columns_and_unequal_compute(self):
        rows = self.rows()
        rows[0]["api_asr"] = ".9"
        with self.assertRaisesRegex(ValueError, "API-derived"):
            score_candidates(rows, self.config, "screen")
        rows = self.rows()
        rows[0]["forward_units_per_item"] = "200"
        with self.assertRaisesRegex(ValueError, "not compute-matched"):
            score_candidates(rows, self.config, "screen")

    def test_augmentation_promotion_covers_additive_geometry_and_combination(self):
        root = Path(self.directory.name) / "augmentation"
        stage = "screen"
        conditions = [
            ({"mode": "none", "sigma": 0.0, "samples": 1, "geometry_mode": "none", "geometry_samples": 1}, .4),
            ({"mode": "gaussian_eot", "sigma": .01, "samples": 2, "geometry_mode": "none", "geometry_samples": 1}, .7),
            ({"mode": "none", "sigma": 0.0, "samples": 1, "geometry_mode": "translation", "geometry_samples": 2}, .8),
            ({"mode": "gaussian_eot", "sigma": .01, "samples": 1, "geometry_mode": "translation", "geometry_samples": 2}, .75),
        ]
        for index, (condition, score) in enumerate(conditions):
            for dataset in ("caption_caltech", "llava_vqa", "receipt_ocr"):
                directory = root / stage / f"trial-{index}-{dataset}"
                directory.mkdir(parents=True)
                value = {
                    **condition, "dataset": dataset, "steps": 100,
                    "budget_label": "equal", "heldout_macro_asr": score,
                    "status": "complete",
                }
                (directory / "trial_result.json").write_text(__import__("json").dumps(value))
        selected = select_augmentations(root, stage, Path(self.directory.name) / "selection.json")
        self.assertEqual(len(selected), 3)
        self.assertTrue(any(row["mode"] != "none" and row["geometry_mode"] == "none" for row in selected))
        self.assertTrue(any(row["mode"] == "none" and row["geometry_mode"] != "none" for row in selected))
        self.assertTrue(any(row["mode"] != "none" and row["geometry_mode"] != "none" for row in selected))


if __name__ == "__main__":
    unittest.main()
