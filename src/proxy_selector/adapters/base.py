"""Strict contract for extracting image-only token representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch

from ..schemas import ImageTokenOutput


class TapValidationError(RuntimeError):
    """Raised when a checkpoint cannot expose the stipulated image-only tap."""


class ImageTokenAdapter(ABC):
    """One frozen model at a time; target adapters never provide gradients."""

    model_id: str
    revision: str | None
    family: str
    role: Literal["proxy", "target"]

    @abstractmethod
    def encode_image_tokens(self, image: torch.Tensor, *, require_grad: bool) -> ImageTokenOutput:
        """Return a validated image-only representation for RGB `[B,3,H,W]`."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources before another large model is loaded."""

    def _validate_output(self, output: ImageTokenOutput, *, require_grad: bool) -> ImageTokenOutput:
        output.validate()
        if self.role == "target" and require_grad:
            raise TapValidationError("Target representations are CKA-only and must not receive gradients.")
        if output.metadata.get("contains_text_tokens") is True:
            raise TapValidationError("Image-token tap includes text tokens.")
        required = {"tap_path", "token_mask_rule", "preprocessing"}
        missing = required.difference(output.metadata)
        if missing:
            raise TapValidationError(f"Tap metadata missing required fields: {sorted(missing)}")
        return output
