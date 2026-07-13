from pathlib import Path
import unittest

from scripts.run_cross_dataset_validation import expected_trial_count, parse_manifest_overrides


class CrossDatasetValidationTests(unittest.TestCase):
    def test_expected_trial_count(self):
        spec = {"stages": {"validation": {
            "datasets": ["caption", "vqa", "receipt"],
            "sets": ["baseline", "candidate"],
            "seeds": [42, 123, 2026],
        }}}
        self.assertEqual(expected_trial_count(spec, "validation"), 18)

    def test_manifest_overrides_require_named_paths(self):
        parsed = parse_manifest_overrides(["caption=/tmp/caption.json", "vqa=/tmp/vqa.json"])
        self.assertEqual(parsed["caption"], Path("/tmp/caption.json"))
        with self.assertRaisesRegex(ValueError, "NAME=PATH"):
            parse_manifest_overrides(["missing-separator"])


if __name__ == "__main__":
    unittest.main()
