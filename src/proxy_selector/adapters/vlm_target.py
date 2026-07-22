"""CKA-only image-token taps for target VLMs.

Target image features are extracted with their native processors under
``torch.no_grad``. These adapters explicitly reject gradients, so they cannot
accidentally become white-box target attacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..representations import global_features, row_normalize
from ..schemas import ImageTokenOutput
from .base import ImageTokenAdapter, TapValidationError


class VLMTargetTokenAdapter(ImageTokenAdapter):
    """Validated pre-text-fusion visual output for Gemma 4 or InternVL."""

    def __init__(self, model_id: str, checkpoint_path: Path, *, family: str, prompt: str, image_keys: tuple[str, ...]) -> None:
        self.model_id, self.revision, self.family = model_id, checkpoint_path.name, family
        self.role: Literal["target"] = "target"
        self.prompt, self.image_keys = prompt, image_keys
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, local_files_only=True, trust_remote_code=family == "InternVL3.5")
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path, local_files_only=True, trust_remote_code=family == "InternVL3.5", torch_dtype=torch.bfloat16
        ).eval().requires_grad_(False).to(self.device)

    def encode_pil(self, image: Image.Image) -> ImageTokenOutput:
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
        kwargs = {key: inputs[key].to(self.device) for key in self.image_keys if key in inputs}
        with torch.no_grad():
            output = self.model.get_image_features(**kwargs)
        tokens = output.last_hidden_state
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 3 or tokens.shape[1] == 0:
            raise TapValidationError("Target visual interface did not yield image-only tokens.")
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        result = ImageTokenOutput(
            global_features=global_features(tokens, mask), local_tokens=row_normalize(tokens), token_mask=mask,
            metadata={"tap_path": "model.get_image_features(...).last_hidden_state", "token_mask_rule": "all native visual-interface tokens are valid",
                      "preprocessing": {"processor": type(self.processor).__name__, "native_processor": True}, "contains_text_tokens": False},
        )
        return self._validate_output(result, require_grad=False)

    def encode_image_tokens(self, image: torch.Tensor, *, require_grad: bool) -> ImageTokenOutput:
        if require_grad:
            raise TapValidationError("Target adapters are CKA-only and reject input gradients.")
        raise TapValidationError("Target native preprocessing is image-file/PIL based; use encode_pil for CKA extraction.")

    def unload(self) -> None:
        self.model.to("cpu")
        del self.model
