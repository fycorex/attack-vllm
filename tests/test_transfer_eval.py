import json
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from api_victims import AnthropicVictim, OpenAIVictim, create_api_victim
from transfer_eval import append_jsonl, conditioned_success, load_replay_config, run_replay, stable_cache_key, summarize


class TransferEvalTests(unittest.TestCase):
    def test_config_and_unsupported_sampling_omitted(self):
        cfg = load_replay_config("configs/transferability_primary.yaml")
        self.assertEqual([m.provider for m in cfg.models], ["openai", "gemini", "anthropic"])
        seen = {}
        def transport(url, headers, payload, timeout):
            seen.update(payload); return {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
        os.environ["ANTHROPIC_API_KEY"] = "test"
        AnthropicVictim({"model_id": "x"}, transport).generate(Image.new("RGB", (2, 2)), "p")
        self.assertNotIn("temperature", seen); self.assertNotIn("top_p", seen); self.assertNotIn("top_k", seen)

    def test_missing_key(self):
        os.environ.pop("GEMINI_API_KEY", None)
        victim = create_api_victim({"provider": "gemini", "model_id": "x"}, lambda *x: {})
        with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
            victim.generate(Image.new("RGB", (2, 2)), "p")

    def test_openai_jpeg_detail_and_browser_headers(self):
        seen = {}
        def transport(url, headers, payload, timeout):
            seen.update(url=url, headers=headers, payload=payload)
            return {"model": "resolved", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        os.environ["OPENAI_API_KEY"] = "test"
        config = {"model_id": "x", "image_format": "jpeg", "jpeg_quality": 80,
                  "image_detail": "low", "browser_headers": True}
        OpenAIVictim(config, transport).generate(Image.new("RGB", (2, 2)), "p")
        image_url = seen["payload"]["messages"][0]["content"][1]["image_url"]
        self.assertTrue(image_url["url"].startswith("data:image/jpeg;base64,/9j/"))
        self.assertEqual(image_url["detail"], "low")
        self.assertIn("Mozilla/5.0", seen["headers"]["User-Agent"])
        self.assertEqual(seen["headers"]["Authorization"], "Bearer test")

    def test_invalid_image_format_fails_before_transport(self):
        os.environ["OPENAI_API_KEY"] = "test"
        victim = OpenAIVictim({"model_id": "x", "image_format": "tiff"}, lambda *args: self.fail("transport called"))
        with self.assertRaisesRegex(ValueError, "Unsupported image_format"):
            victim.generate(Image.new("RGB", (2, 2)), "p")

    def test_cache_stability(self):
        a = stable_cache_key("p", "m", "q", "i", {"b": 2, "a": 1})
        self.assertEqual(a, stable_cache_key("p", "m", "q", "i", {"a": 1, "b": 2}))
        self.assertNotEqual(a, stable_cache_key("p", "m", "other", "i", {"a": 1, "b": 2}))

    def test_conditioned_and_failed_denominator(self):
        self.assertTrue(conditioned_success(False, True)); self.assertFalse(conditioned_success(True, True))
        base = {"provider": "p", "model_id": "m", "item_id": "1", "refusal": False, "source_present": False}
        rows = [{**base, "condition": "clean", "target_success": False, "error": None},
                {**base, "condition": "adversarial", "target_success": False, "error": {"type": "x"}}]
        result = summarize(rows, 10)
        self.assertEqual(result["paired_item_count"], 1); self.assertEqual(result["valid_pair_count"], 0)
        self.assertEqual(result["api_failure_rate"], .5)

    def test_same_item_id_from_two_candidates_remains_two_pairs(self):
        base = {"provider": "p", "model_id": "m", "item_id": "same", "refusal": False,
                "source_present": False, "error": None}
        rows = []
        for sample in ("candidate-a", "candidate-b"):
            rows.extend([{**base, "evaluation_sample_id": sample, "condition": "clean", "target_success": False},
                         {**base, "evaluation_sample_id": sample, "condition": "adversarial", "target_success": True}])
        result = summarize(rows, 10)
        self.assertEqual(result["paired_item_count"], 2)
        self.assertEqual(result["conditioned_asr"], 1.0)

    def test_atomic_append(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.jsonl"; append_jsonl(path, {"x": 1}); append_jsonl(path, {"x": 2})
            self.assertEqual([json.loads(x)["x"] for x in path.read_text().splitlines()], [1, 2])

    def test_no_opt_in_is_dry_run_and_request_equivalence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); item = root / "source" / "item"; item.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(item / "clean.png"); Image.new("RGB", (2, 2)).save(item / "adversarial.png")
            (item / "metrics.json").write_text(json.dumps({"item_id": "x", "target_label": "cat", "source_label": "dog", "target_keywords": ["cat"], "source_keywords": ["dog"]}))
            config = root / "c.yaml"; config.write_text("experiment_name: x\nmodels:\n  - provider: openai\n    model_id: fake\n")
            result = run_replay(config, [root / "source"], root / "results")
            self.assertTrue(result["dry_run"]); self.assertEqual(result["evaluation_records"], 2)
            self.assertEqual(result["estimated_requests"], 1)

    def test_duplicate_clean_request_is_reused_without_collapsing_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for candidate, color in (("a", "red"), ("b", "blue")):
                item = root / candidate / "item"; item.mkdir(parents=True)
                Image.new("RGB", (2, 2), "white").save(item / "clean.png")
                Image.new("RGB", (2, 2), color).save(item / "adversarial.png")
                (item / "metrics.json").write_text(json.dumps({"item_id": "same", "target_label": "cat",
                    "source_label": "dog", "target_keywords": ["cat"], "source_keywords": ["dog"]}))
            config = root / "c.yaml"
            config.write_text("experiment_name: x\nmodels:\n  - provider: openai\n    model_id: fake\n    max_retries: 0\n")
            calls = []
            def transport(url, headers, payload, timeout):
                calls.append(payload)
                return {"model": "fake", "choices": [{"message": {"content": "cat"}, "finish_reason": "stop"}]}
            os.environ["OPENAI_API_KEY"] = "test"
            result = run_replay(config, [root / "a", root / "b"], root / "results",
                                allow_real_api=True, transports={"openai": transport})
            records = [json.loads(line) for line in (root / "results" / "requests.jsonl").read_text().splitlines()]
            summary = json.loads((root / "results" / "summary_combined.json").read_text())
            self.assertEqual(result["evaluation_records"], 4)
            self.assertEqual(result["estimated_requests"], 3)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(records), 4)
            self.assertEqual(sum(row["response_reused"] for row in records), 1)
            self.assertEqual(summary["paired_item_count"], 2)
            self.assertEqual(len(summary["candidate_models"]), 2)
            self.assertTrue((root / "results" / "summary_by_candidate_model.csv").is_file())


if __name__ == "__main__": unittest.main()
