import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_geometric_gated_pipeline import require_valid_smoke


class GeometricGateTests(unittest.TestCase):
    def test_requires_all_expected_smoke_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            path.write_text(json.dumps({"all_valid": True, "trials": 6, "valid_trials": 6}))
            self.assertEqual(require_valid_smoke(path, 6)["valid_trials"], 6)
            with self.assertRaisesRegex(RuntimeError, "gate failed"):
                require_valid_smoke(path, 7)

    def test_missing_smoke_report_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            require_valid_smoke(Path("/definitely/missing/geometric-validation.json"), 6)


if __name__ == "__main__":
    unittest.main()
