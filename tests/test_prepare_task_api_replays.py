import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from scripts.prepare_task_api_replays import parse_dataset_configs, prepare_replays


class PrepareTaskApiReplaysTests(unittest.TestCase):
    def test_prepares_task_specific_baseline_candidate_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for method, asr in (("single_reference", .2), ("two_homogeneous", .7)):
                for seed in (42, 123, 2026):
                    trial = root / method / str(seed)
                    item = trial / "attack" / "item_0"
                    item.mkdir(parents=True)
                    (item / "metrics.json").write_text(json.dumps({
                        "item_id": "item_0", "target_keywords": ["target"],
                        "source_keywords": ["source"],
                    }))
                    Image.new("RGB", (2, 2), (seed % 255, 0, 0)).save(item / "clean.png")
                    Image.new("RGB", (2, 2), (seed % 255, 1, 0)).save(item / "adversarial.png")
                    rows.append({
                        "stage": "validation", "budget_mode": "equal_steps", "dataset": "caption",
                        "surrogate_set": method, "seed": seed, "target": "heldout", "asr": asr,
                        "mean_margin_gain": asr / 10, "items": 1, "valid_items": 1,
                        "missing_items": 0, "directory": str(trial),
                    })
            analysis = root / "analysis.csv"
            with analysis.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            config = root / "caption.yaml"
            config.write_text(yaml.safe_dump({
                "experiment_name": "dry", "task_type": "caption",
                "models": [{"provider": "openai", "model_id": "vision-test", "api_key_env": "UNSET_TEST_KEY"}],
            }))

            result = prepare_replays(
                analysis_csv=analysis, output=root / "plan", dataset_configs={"caption": config},
                stage="validation", budget_mode="equal_steps", baseline="single_reference",
                top_k=1, minimum_seeds=3, minimum_targets=1,
            )

            self.assertFalse(result["real_api"])
            self.assertEqual(result["datasets"]["caption"]["candidates"],
                             ["single_reference", "two_homogeneous"])
            self.assertEqual(result["datasets"]["caption"]["evaluation_records"], 12)
            self.assertTrue((root / "plan" / "api_replay_plan.json").is_file())

    def test_dataset_config_syntax(self):
        self.assertEqual(parse_dataset_configs(["vqa=/tmp/vqa.yaml"]), {"vqa": Path("/tmp/vqa.yaml")})
        with self.assertRaisesRegex(ValueError, "DATASET=YAML"):
            parse_dataset_configs(["broken"])


if __name__ == "__main__":
    unittest.main()
