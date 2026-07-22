"""Differentiable Hugging Face vision-only adapters used by CLIP and SigLIP2."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from ..representations import global_features, row_normalize
from ..schemas import ImageTokenOutput
from .base import ImageTokenAdapter, TapValidationError


class HFVisionTokenAdapter(ImageTokenAdapter):
    """Extract final vision-encoder tokens without ever invoking text encoders."""

    def __init__(
        self,
        model_id: str,
        checkpoint_path: Path,
        *,
        family: str,
        drop_first_token: bool,
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.revision = checkpoint_path.name
        self.family = family
        self.role: Literal["proxy"] = "proxy"
        self.drop_first_token = drop_first_token
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(checkpoint_path, local_files_only=True)
        # Keep text access on the very same checkpoint.  It is only used by the
        # positive-control direction loss; no second surrogate is introduced.
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(checkpoint_path, local_files_only=True, torch_dtype=torch.bfloat16)
        self.model.eval().requires_grad_(False).to(self.device)
        size = self.processor.size
        self.height = int(size.get("height", size.get("shortest_edge", size.get("width", 224))))
        self.width = int(size.get("width", self.height))
        self.mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB tensor [B, 3, H, W].")
        pixels = F.interpolate(image.to(self.device), size=(self.height, self.width), mode="bilinear", align_corners=False, antialias=True)
        return (pixels - self.mean.to(pixels.dtype)) / self.std.to(pixels.dtype)

    def encode_image_tokens(self, image: torch.Tensor, *, require_grad: bool) -> ImageTokenOutput:
        return self.encode_image_layers(image, require_grad=require_grad)["final"]

    def _output_from_tokens(self, tokens: torch.Tensor, *, tap_path: str, fraction: float) -> ImageTokenOutput:
        if self.drop_first_token:
            tokens = tokens[:, 1:, :]
        if tokens.shape[1] == 0:
            raise TapValidationError("Vision output has no patch tokens after special-token removal.")
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return ImageTokenOutput(
            global_features=global_features(tokens, mask),
            local_tokens=row_normalize(tokens),
            token_mask=mask,
            metadata={
                "tap_path": tap_path,
                "layer_fraction": fraction,
                "token_mask_rule": "all vision patch tokens are valid",
                "preprocessing": {"size": [self.height, self.width], "mean": self.processor.image_mean, "std": self.processor.image_std},
                "contains_text_tokens": False,
                "dropped_first_token": self.drop_first_token,
            },
        )

    def encode_image_layers(self, image: torch.Tensor, *, require_grad: bool) -> dict[str, ImageTokenOutput]:
        """Return quarter/half/three-quarter/final visual-token taps.

        The hidden-state request is deliberately made at the vision encoder, so
        all returned tensors remain image-only and differentiable w.r.t. the
        original RGB input.
        """
        layers, _ = self.encode_hierarchical_semantic(image, require_grad=require_grad)
        return layers

    def encode_hierarchical_semantic(
        self, image: torch.Tensor, *, require_grad: bool
    ) -> tuple[dict[str, ImageTokenOutput], torch.Tensor]:
        """Extract hierarchy and shared image/text semantic vector in one pass."""
        pixels = self._preprocess(image)
        context = torch.enable_grad() if require_grad else torch.no_grad()
        with context:
            vision = self.model.vision_model(pixel_values=pixels, output_hidden_states=True, return_dict=True)
            states = vision.hidden_states
            if not states:
                raise TapValidationError("Vision model did not return hidden states for hierarchical attack.")
            fractions = {"25": 0.25, "50": 0.50, "75": 0.75, "final": 1.0}
            outputs: dict[str, ImageTokenOutput] = {}
            for label, fraction in fractions.items():
                index = max(1, min(len(states) - 1, round((len(states) - 1) * fraction)))
                outputs[label] = self._validate_output(
                    self._output_from_tokens(
                        states[index],
                        tap_path=f"model.vision_model(..., output_hidden_states=True).hidden_states[{index}]",
                        fraction=fraction,
                    ),
                    require_grad=require_grad,
                )
            pooled = vision.pooler_output
            projection = getattr(self.model, "visual_projection", None)
            if projection is not None:
                pooled = projection(pooled)
        return outputs, F.normalize(pooled.float(), dim=-1)

    def semantic_global(self, image: torch.Tensor, *, require_grad: bool) -> torch.Tensor:
        """Shared image/text embedding used for UnivIntruder-style direction."""
        _, semantic = self.encode_hierarchical_semantic(image, require_grad=require_grad)
        return semantic

    def encode_text_concepts(self, texts: list[str]) -> torch.Tensor:
        """Same-checkpoint text embeddings in the shared semantic space."""
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            text = self.model.get_text_features(**encoded)
            # transformers 5 returns BaseModelOutputWithPooling for several
            # CLIP/SigLIP variants rather than the projected tensor.
            if hasattr(text, "pooler_output"):
                text = text.pooler_output
                projection = getattr(self.model, "text_projection", None)
                if projection is not None:
                    text = projection(text)
            return F.normalize(text.float(), dim=-1)

    def unload(self) -> None:
        self.model.to("cpu")
        del self.model
