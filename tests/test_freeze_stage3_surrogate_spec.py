import csv
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.freeze_stage3_surrogate_spec import audit_manifest, freeze, select_methods


FIELDS = ["stage", "budget_mode", "dataset", "surrogate_set", "seed", "target", "asr",
          "mean_margin_gain", "items", "valid_items", "missing_items", "directory"]


class FreezeStage3Tests(unittest.TestCase):
    def rows(self):
        rows = []
        for method, score in (("single_reference", .2), ("two", .5), ("four", .4)):
            for seed in (42, 123, 2026):
                for target in ("a", "b"):
                    rows.append({"stage": "stage2_equal_forwards", "budget_mode": "equal_forwards",
                                 "dataset": "caption", "surrogate_set": method, "seed": str(seed),
                                 "target": target, "asr": str(score), "mean_margin_gain": str(score / 10),
                                 "items": "20", "valid_items": "20", "missing_items": "0",
                                 "directory": f"/tmp/{method}/{seed}"})
        return rows

    def test_selects_baseline_and_best_complete_alternative(self):
        rows = self.rows()
        rows = [row for row in rows if not (row["surrogate_set"] == "four" and row["seed"] == "2026" and row["target"] == "b")]
        selected = select_methods(rows, stage="stage2_equal_forwards", budget_mode="equal_forwards",
                                  dataset="caption", baseline="single_reference", top_k=2, minimum_seeds=3)
        self.assertEqual([row["surrogate_set"] for row in selected], ["single_reference", "two"])

    def test_writes_hashed_stage3_spec(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); analysis = root / "analysis.csv"; base = root / "base.yaml"; output = root / "frozen.yaml"
            with analysis.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(self.rows())
            images = []
            for index in range(2):
                image = root / f"{index}.png"; image.write_bytes(b"image"); images.append(str(image))
            manifest = root / "manifest.json"
            manifest.write_text(__import__("json").dumps({"items": [{"image_path": image} for image in images]}))
            base.write_text(yaml.safe_dump({"datasets": {"caption": {"manifest": str(manifest)}},
                "stages": {"stage3_cross_dataset": {"sets": ["placeholder"], "datasets": ["caption"], "items": 2}}}))
            evidence = freeze(base, analysis, output, selection_dataset="caption", top_k=1, stage3_items=2)
            value = yaml.safe_load(output.read_text())
            self.assertEqual(value["stages"]["stage3_cross_dataset"]["sets"], ["single_reference", "two"])
            self.assertEqual(value["frozen_stage3_selection"]["freeze_hash"], evidence["freeze_hash"])
            self.assertFalse(evidence["api_results_used_for_selection"])
            self.assertEqual(evidence["dataset_manifests"]["caption"]["unique_selected_source_images"], 2)

    def test_can_freeze_equal_forward_validation_stage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); analysis = root / "analysis.csv"; base = root / "base.yaml"; output = root / "frozen.yaml"
            with analysis.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(self.rows())
            images = []
            for index in range(2):
                image = root / f"{index}.png"; image.write_bytes(b"image"); images.append(str(image))
            manifest = root / "manifest.json"
            manifest.write_text(__import__("json").dumps({"items": [{"image_path": image} for image in images]}))
            base.write_text(yaml.safe_dump({"datasets": {"caption": {"manifest": str(manifest)}},
                "stages": {"validation_equal_forward": {"sets": ["placeholder"], "datasets": ["caption"],
                                                            "items": 2, "budget_mode": "equal_forwards"}}}))
            freeze(base, analysis, output, selection_dataset="caption", top_k=1, stage3_items=2,
                   target_stage="validation_equal_forward")
            value = yaml.safe_load(output.read_text())
            self.assertEqual(value["stages"]["validation_equal_forward"]["sets"], ["single_reference", "two"])
            self.assertEqual(value["frozen_stage3_selection"]["target_stage"], "validation_equal_forward")

    def test_manifest_audit_rejects_repeated_source_images(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); image = root / "x.png"; image.write_bytes(b"image")
            manifest = root / "manifest.json"
            manifest.write_text(__import__("json").dumps({"items": [{"image_path": str(image)}, {"image_path": str(image)}]}))
            with self.assertRaisesRegex(ValueError, "unique source image"):
                audit_manifest(manifest, 2)


if __name__ == "__main__":
    unittest.main()
