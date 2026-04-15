from __future__ import annotations

from pathlib import Path
import re

from PIL import Image
import torch

from config import CaptionVictimConfig
from data import AttackItem


class CaptionVictim:
    def __init__(self, config: CaptionVictimConfig, cache_dir: str | Path | None = None):
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
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Caption victim requires the transformers package. Install project requirements first with: pip install -r requirements.txt"
            ) from exc

        cache_dir = str(self.cache_dir) if self.cache_dir is not None else None
        self._processor = BlipProcessor.from_pretrained(self.config.model_name, cache_dir=cache_dir)
        model = BlipForConditionalGeneration.from_pretrained(
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

    def caption_image(self, image: Image.Image) -> str:
        self._load()
        prompt = self.config.prompt
        if prompt:
            inputs = self._processor(image, prompt, return_tensors="pt")
        else:
            inputs = self._processor(image, return_tensors="pt")
        inputs = self._move_inputs(dict(inputs))
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
            )
        caption = self._processor.decode(output[0], skip_special_tokens=True).strip().lower()
        return re.sub(r"\s+", " ", caption)

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        text = text.lower()
        for keyword in keywords:
            pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
            if re.search(pattern, text):
                return True
        return False

    def evaluate(self, clean_image: Image.Image, adversarial_image: Image.Image, item: AttackItem) -> dict:
        clean_caption = self.caption_image(clean_image)
        adversarial_caption = self.caption_image(adversarial_image)
        clean_mentions_source = self._contains_any(clean_caption, item.source_keywords)
        clean_mentions_target = self._contains_any(clean_caption, item.target_keywords)
        adversarial_mentions_source = self._contains_any(adversarial_caption, item.source_keywords)
        adversarial_mentions_target = self._contains_any(adversarial_caption, item.target_keywords)
        success = adversarial_mentions_target and (
            not self.config.require_source_absent or not adversarial_mentions_source
        )
        result = {
            "model_name": self.config.model_name,
            "clean_caption": clean_caption,
            "adversarial_caption": adversarial_caption,
            "clean_mentions_source": clean_mentions_source,
            "clean_mentions_target": clean_mentions_target,
            "adversarial_mentions_source": adversarial_mentions_source,
            "adversarial_mentions_target": adversarial_mentions_target,
            "caption_success": success,
        }
        if self.config.sequential_loading:
            self.unload()
        return result
