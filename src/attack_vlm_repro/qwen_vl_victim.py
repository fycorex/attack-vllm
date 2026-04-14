"""HuggingFace QwenVL victim as fallback when ollama is unstable."""
from __future__ import annotations

from pathlib import Path
import re

from PIL import Image
import torch

from .config import HuggingFaceQwenVLConfig
from .data import AttackItem


class QwenVLVictim:
    """Victim model using HuggingFace QwenVL transformers."""

    def __init__(self, config: HuggingFaceQwenVLConfig, cache_dir: str | Path | None = None):
        self.config = config
        self._processor = None
        self._model = None
        self.device = config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" and config.use_fp16 else torch.float32
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "QwenVL victim requires the transformers package. "
                "Install with: pip install transformers accelerate"
            ) from exc

        cache_dir = str(self.cache_dir) if self.cache_dir is not None else None
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=self.dtype,
            cache_dir=cache_dir,
            device_map=self.device,
        )
        self._model = model.eval()

    def unload(self) -> None:
        self._processor = None
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        """Generate response from image and prompt."""
        self._load()

        # Build messages for QwenVL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
            )

        # Decode output
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        output_text = self._processor.decode(generated_ids, skip_special_tokens=True)
        return self._normalize_text(output_text)

    def evaluate(self, clean_image: Image.Image, adversarial_image: Image.Image, item: AttackItem) -> dict:
        """Evaluate attack success against QwenVL."""
        question = item.question or self.config.question_fallback

        try:
            clean_output = self.generate(clean_image, question)
        except Exception as e:
            return {
                "model_name": self.config.model_name,
                "qwen_vl_success": False,
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
                "qwen_vl_success": False,
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

        result = {
            "model_name": self.config.model_name,
            "question": question,
            "clean_output": clean_output,
            "adversarial_output": adversarial_output,
            "clean_mentions_source": clean_mentions_source,
            "clean_mentions_target": clean_mentions_target,
            "adversarial_mentions_source": adversarial_mentions_source,
            "adversarial_mentions_target": adversarial_mentions_target,
            "qwen_vl_success": success,
            "evaluation_failed": False,
        }

        if self.config.sequential_loading:
            self.unload()

        return result
