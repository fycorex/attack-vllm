import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.validate_geometric_stage import finite_tree, image_linf


class GeometricValidationTests(unittest.TestCase):
    def test_finite_tree(self):
        self.assertTrue(finite_tree({"x": [1.0, {"y": 2}]}))
        self.assertFalse(finite_tree({"x": float("nan")}))
        self.assertFalse(finite_tree([math.inf]))

    def test_saved_image_linf(self):
        with tempfile.TemporaryDirectory() as td:
            clean, adversarial = Path(td) / "clean.png", Path(td) / "adversarial.png"
            Image.new("RGB", (2, 2), (100, 100, 100)).save(clean)
            Image.new("RGB", (2, 2), (102, 100, 100)).save(adversarial)
            self.assertAlmostEqual(image_linf(clean, adversarial), 2 / 255, places=6)


if __name__ == "__main__":
    unittest.main()
