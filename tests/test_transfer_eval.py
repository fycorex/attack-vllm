import json
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from api_victims import AnthropicVictim, create_api_victim
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
            self.assertTrue(result["dry_run"]); self.assertEqual(result["estimated_requests"], 2)


if __name__ == "__main__": unittest.main()
