"""Ollama VLM victim for local QwenVL evaluation."""
from __future__ import annotations

from base64 import b64encode
from io import BytesIO
import json
import re
import time
from urllib import error as urlerror
from urllib import request as urlrequest

from PIL import Image

from config import OllamaVictimConfig
from data import AttackItem


class OllamaVictim:
    """Victim model using local ollama VLM (e.g., QwenVL)."""

    def __init__(self, config: OllamaVictimConfig):
        self.config = config

    def health_check(self) -> dict:
        """Check ollama service and model availability."""
        try:
            req = urlrequest.Request(
                f"{self.config.base_url}/api/tags",
                method="GET"
            )
            with urlrequest.urlopen(req, timeout=self.config.health_check_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models", [])
            model_names = [m["name"] for m in models]
            return {
                "healthy": self.config.model_name in model_names,
                "available_models": model_names,
                "error": None
            }
        except Exception as e:
            return {
                "healthy": False,
                "available_models": [],
                "error": str(e)
            }

    def ensure_loaded(self) -> bool:
        """Pre-load model with keep_alive to reduce cold-start crashes."""
        try:
            payload = json.dumps({
                "model": self.config.model_name,
                "keep_alive": "5m",
                "prompt": ""
            }).encode("utf-8")
            req = urlrequest.Request(
                f"{self.config.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            urlrequest.urlopen(req, timeout=30.0)
            return True
        except Exception:
            return False

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        normalized = text.lower()
        for keyword in keywords:
            pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
            if re.search(pattern, normalized):
                return True
        return False

    def generate(self, image: Image.Image, prompt: str) -> str:
        """Generate response using native ollama API with image."""
        image_b64 = self._encode_image(image)

        payload = json.dumps({
            "model": self.config.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }).encode("utf-8")

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                req = urlrequest.Request(
                    f"{self.config.base_url}/{self.config.api_endpoint}",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urlrequest.urlopen(req, timeout=self.config.request_timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                return self._normalize_text(result.get("response", ""))
            except urlerror.HTTPError as e:
                last_error = RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_backoff * (2 ** attempt))
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_backoff * (2 ** attempt))

        raise RuntimeError(f"Ollama request failed after {self.config.max_retries} retries: {last_error}")

    def evaluate(self, clean_image: Image.Image, adversarial_image: Image.Image, item: AttackItem) -> dict:
        """Evaluate attack success against ollama VLM."""
        # Health check first
        health = self.health_check()
        if not health["healthy"]:
            return {
                "model_name": self.config.model_name,
                "ollama_success": False,
                "evaluation_failed": True,
                "error": f"Health check failed: {health.get('error', 'model not available')}",
                "health": health
            }

        question = item.question or self.config.question_fallback

        try:
            clean_output = self.generate(clean_image, question)
        except Exception as e:
            return {
                "model_name": self.config.model_name,
                "ollama_success": False,
                "evaluation_failed": True,
                "error": f"Clean image generation failed: {e}",
            }

        try:
            adversarial_output = self.generate(adversarial_image, question)
        except Exception as e:
            return {
                "model_name": self.config.model_name,
                "question": question,
                "clean_output": clean_output,
                "ollama_success": False,
                "evaluation_failed": True,
                "error": f"Adversarial image generation failed: {e}",
            }

        # Check success based on keywords
        clean_mentions_source = self._contains_any(clean_output, item.source_answer_keywords)
        clean_mentions_target = self._contains_any(clean_output, item.target_answer_keywords)
        adversarial_mentions_source = self._contains_any(adversarial_output, item.source_answer_keywords)
        adversarial_mentions_target = self._contains_any(adversarial_output, item.target_answer_keywords)

        success = adversarial_mentions_target and (
            not self.config.require_source_absent or not adversarial_mentions_source
        )

        return {
            "model_name": self.config.model_name,
            "task_type": self.config.task_type,
            "question": question,
            "clean_output": clean_output,
            "adversarial_output": adversarial_output,
            "clean_mentions_source": clean_mentions_source,
            "clean_mentions_target": clean_mentions_target,
            "adversarial_mentions_source": adversarial_mentions_source,
            "adversarial_mentions_target": adversarial_mentions_target,
            "ollama_success": success,
            "evaluation_failed": False,
        }

    def unload(self) -> None:
        """Cleanup - no persistent resources."""
        pass
