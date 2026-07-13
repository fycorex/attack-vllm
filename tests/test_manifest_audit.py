import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.audit_attack_manifest import audit_manifest
from scripts.prepare_trainingdatapro_receipts_text import select_qa_rows


class ManifestAuditTests(unittest.TestCase):
    def test_single_receipt_questions_are_balanced_by_image(self):
        rows = [{"question_type": "store"}, {"question_type": "total"}]
        self.assertEqual(select_qa_rows(rows, 1, 0)[0]["question_type"], "store")
        self.assertEqual(select_qa_rows(rows, 1, 1)[0]["question_type"], "total")

    def test_duplicate_source_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); image = root / "image.png"; example = root / "example.png"
            Image.new("RGB", (2, 2), "white").save(image); Image.new("RGB", (2, 2), "black").save(example)
            items = []
            for index in range(2):
                items.append({"id": f"x{index}", "image_path": str(image), "source_label": "a", "target_label": "b",
                    "question": "q", "target_answer_text": "b", "positive_image_paths": [str(example)],
                    "negative_image_paths": [str(example)], "metadata": {"source_image_id": f"s{index}", "target_image_id": f"t{index}"}})
            manifest = root / "manifest.json"; manifest.write_text(json.dumps({"dataset_name": "x", "items": items}))
            report = audit_manifest(manifest, require_unique_sources=True)
            self.assertFalse(report["valid"])
            self.assertEqual(report["unique_source_images"], 1)


if __name__ == "__main__":
    unittest.main()
