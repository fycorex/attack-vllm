from __future__ import annotations

from pathlib import Path
import re
import shutil

from PIL import Image
import torch

from .config import OCRVictimConfig
from .data import AttackItem


class OCRVictim:
    def __init__(self, config: OCRVictimConfig, cache_dir: str | Path | None = None):
        self.config = config
        self._processor = None
        self._model = None
        self.device = config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" and config.use_fp16 else torch.float32
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def backend(self) -> str:
        return (self.config.backend or "tesseract").strip().lower()

    def _load(self) -> None:
        if self.backend == "tesseract":
            self._validate_tesseract_runtime()
            return
        if self._processor is not None and self._model is not None:
            return
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError(
                "TrOCR backend requires the transformers package. Install project requirements first."
            ) from exc

        cache_dir = str(self.cache_dir) if self.cache_dir is not None else None
        self._processor = TrOCRProcessor.from_pretrained(self.config.model_name, cache_dir=cache_dir)
        model = VisionEncoderDecoderModel.from_pretrained(
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

    @staticmethod
    def _validate_tesseract_runtime() -> None:
        if shutil.which("tesseract") is None:
            raise RuntimeError(
                "Tesseract backend requested but the 'tesseract' binary was not found on PATH. "
                "Install it first, for example: sudo apt-get update && sudo apt-get install -y tesseract-ocr"
            )
        try:
            import pytesseract  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Tesseract backend requested but the Python package 'pytesseract' is not installed. "
                "Install project requirements first with: pip install -r requirements.txt"
            ) from exc

    def _recognize_with_tesseract(self, image: Image.Image) -> str:
        self._validate_tesseract_runtime()
        import pytesseract

        config = f"--psm {self.config.tesseract_psm}"
        text = pytesseract.image_to_string(image, config=config)
        return self._normalize_text(text)

    def _recognize_with_trocr(self, image: Image.Image) -> str:
        self._load()
        pixel_values = self._processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
            )
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self._normalize_text(text)

    def recognize_text(self, image: Image.Image) -> str:
        if self.backend == "tesseract":
            return self._recognize_with_tesseract(image)
        if self.backend == "trocr":
            return self._recognize_with_trocr(image)
        raise ValueError(f"Unsupported OCR backend: {self.config.backend}")

    def evaluate(self, clean_image: Image.Image, adversarial_image: Image.Image, item: AttackItem) -> dict:
        clean_text = self.recognize_text(clean_image)
        adversarial_text = self.recognize_text(adversarial_image)
        clean_mentions_source = self._contains_any(clean_text, item.source_text_keywords)
        clean_mentions_target = self._contains_any(clean_text, item.target_text_keywords)
        adversarial_mentions_source = self._contains_any(adversarial_text, item.source_text_keywords)
        adversarial_mentions_target = self._contains_any(adversarial_text, item.target_text_keywords)
        success = adversarial_mentions_target and (
            not self.config.require_source_absent or not adversarial_mentions_source
        )
        result = {
            "backend": self.backend,
            "model_name": self.config.model_name,
            "clean_text": clean_text,
            "adversarial_text": adversarial_text,
            "clean_mentions_source": clean_mentions_source,
            "clean_mentions_target": clean_mentions_target,
            "adversarial_mentions_source": adversarial_mentions_source,
            "adversarial_mentions_target": adversarial_mentions_target,
            "ocr_success": success,
        }
        if self.config.sequential_loading:
            self.unload()
        return result
