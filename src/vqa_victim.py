from __future__ import annotations

from pathlib import Path
import re

from PIL import Image
import torch

from config import VQAVictimConfig
from data import AttackItem


class VQAVictim:
    def __init__(self, config: VQAVictimConfig, cache_dir: str | Path | None = None):
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
            from transformers import BlipForQuestionAnswering, BlipProcessor
        except ImportError as exc:
            raise RuntimeError(
                "VQA victim requires the transformers package. Install project requirements first with: pip install -r requirements.txt"
            ) from exc

        cache_dir = str(self.cache_dir) if self.cache_dir is not None else None
        self._processor = BlipProcessor.from_pretrained(self.config.model_name, cache_dir=cache_dir)
        model = BlipForQuestionAnswering.from_pretrained(
            self.config.model_name,
            torch_dtype=self.dtype,
            cache_dir=cache_dir,
        )
        self._model = model.to(self.device)
        self._model.eval()

    def unload(self) -> None:
        self._processor = None
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        moved = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                moved[key] = value
                continue
            if value.dtype.is_floating_point:
                moved[key] = value.to(self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(self.device)
        return moved

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

    def answer_question(self, image: Image.Image, question: str) -> str:
        self._load()
        inputs = self._processor(image, question, return_tensors="pt")
        inputs = self._move_inputs(dict(inputs))
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
            )
        answer = self._processor.decode(output[0], skip_special_tokens=True)
        return self._normalize_text(answer)

    def evaluate(self, clean_image: Image.Image, adversarial_image: Image.Image, item: AttackItem) -> dict:
        question = item.question or self.config.question_fallback
        clean_answer = self.answer_question(clean_image, question)
        adversarial_answer = self.answer_question(adversarial_image, question)
        clean_mentions_source = self._contains_any(clean_answer, item.source_answer_keywords)
        clean_mentions_target = self._contains_any(clean_answer, item.target_answer_keywords)
        adversarial_mentions_source = self._contains_any(adversarial_answer, item.source_answer_keywords)
        adversarial_mentions_target = self._contains_any(adversarial_answer, item.target_answer_keywords)
        success = adversarial_mentions_target and (
            not self.config.require_source_absent or not adversarial_mentions_source
        )
        result = {
            "model_name": self.config.model_name,
            "question": question,
            "clean_answer": clean_answer,
            "adversarial_answer": adversarial_answer,
            "clean_mentions_source": clean_mentions_source,
            "clean_mentions_target": clean_mentions_target,
            "adversarial_mentions_source": adversarial_mentions_source,
            "adversarial_mentions_target": adversarial_mentions_target,
            "vqa_success": success,
        }
        if self.config.sequential_loading:
            self.unload()
        return result
