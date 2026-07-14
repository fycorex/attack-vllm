import unittest

from scripts.build_final_api_rubric import item_rows, rubric_rows


class FinalApiRubricTests(unittest.TestCase):
    def test_groups_frozen_candidate_without_selecting(self):
        records = []
        for condition, target_success in (("clean", False), ("adversarial", True)):
            records.append({
                "task_type": "caption", "candidate_method": "two_cross_objective",
                "candidate_selection_role": "heldout_selected", "provider": "openai",
                "model_id": "gpt-4o", "evaluation_sample_id": "sample-1", "item_id": "item-1",
                "condition": condition, "target_success": target_success,
                "source_present": condition == "clean", "refusal": False, "error": None,
                "image_sha256": "same-source",
            })
        rows = rubric_rows(records, bootstrap_samples=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["conditioned_asr"], 1.0)
        self.assertEqual(rows[0]["model_id"], "gpt-4o")
        paired = item_rows(records)
        self.assertTrue(paired[0]["conditioned_success"])
        self.assertTrue(paired[0]["pair_valid"])


if __name__ == "__main__":
    unittest.main()
