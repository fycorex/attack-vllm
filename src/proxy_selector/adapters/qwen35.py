"""Differentiable image-token adapter for Qwen3.5-VL checkpoints.

The public processor serializes an image into flattened temporal patches.  This
module reproduces that image-only path with Torch operations so the attack
tensor remains connected to the visual merger output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..representations import global_features, row_normalize
from ..schemas import ImageTokenOutput
from .base import ImageTokenAdapter, TapValidationError


@dataclass(frozen=True)
class QwenAnswerTarget:
    """Static text-side tensors for one question/answer loss.

    The dummy image is used only while tokenizing the chat template so that its
    image-pad count matches the source resolution.  The differentiable attack
    image is always supplied separately through ``pixel_values``.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mm_token_type_ids: torch.Tensor
    labels: torch.Tensor


def qwen_smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Match the Qwen image processor's deterministic output-size policy."""
    if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError("Qwen image aspect ratio must be finite and at most 200.")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


class Qwen35TokenAdapter(ImageTokenAdapter):
    """Qwen merger-output image tokens, before language/text-token fusion."""

    def __init__(self, model_id: str, checkpoint_path: Path, *, device: str = "cuda") -> None:
        self.model_id = model_id
        self.revision = checkpoint_path.name
        self.family = "Qwen3.5"
        self.role: Literal["proxy"] = "proxy"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path, local_files_only=True, torch_dtype=torch.bfloat16
        ).eval().requires_grad_(False).to(self.device)
        image_processor = self.processor.image_processor
        vision = self.model.config.vision_config
        self.patch_size = int(vision.patch_size)
        self.temporal_patch_size = int(vision.temporal_patch_size)
        self.merge_size = int(vision.spatial_merge_size)
        self.factor = self.patch_size * self.merge_size
        self.min_pixels = int(image_processor.size.shortest_edge)
        self.max_pixels = int(image_processor.size.longest_edge)
        self.mean = torch.tensor(image_processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(image_processor.image_std, device=self.device).view(1, 3, 1, 1)

    def _pixel_values(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB tensor [B, 3, H, W].")
        # A Qwen get_image_features call accepts a concatenated image sequence;
        # batching arbitrary original sizes would require a ragged patch tensor.
        if image.shape[0] != 1:
            raise ValueError("Qwen35TokenAdapter currently accepts one image per call.")
        height, width = image.shape[-2:]
        resized_height, resized_width = qwen_smart_resize(
            height,
            width,
            factor=self.factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        pixels = F.interpolate(
            image.to(self.device),
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.mean.to(pixels.dtype)) / self.std.to(pixels.dtype)
        batch, channels, _, _ = pixels.shape
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = pixels.reshape(
            batch,
            channels,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        ).permute(0, 2, 5, 3, 6, 1, 4, 7)
        flattened = patches.unsqueeze(6).expand(
            -1, -1, -1, -1, -1, -1, self.temporal_patch_size, -1, -1
        ).reshape(batch * grid_h * grid_w, channels * self.temporal_patch_size * self.patch_size**2)
        grid = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long, device=self.device)
        return flattened, grid

    def encode_image_tokens(self, image: torch.Tensor, *, require_grad: bool) -> ImageTokenOutput:
        pixel_values, grid = self._pixel_values(image)
        context = torch.enable_grad() if require_grad else torch.no_grad()
        with context:
            vision = self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=grid)
            tokens = vision.pooler_output
            if isinstance(tokens, tuple):
                if len(tokens) != 1:
                    raise TapValidationError("Expected one Qwen image-token sequence per input image.")
                tokens = tokens[0]
            if tokens.ndim != 2 or tokens.shape[0] == 0:
                raise TapValidationError("Qwen visual merger did not return an image-token sequence.")
            tokens = tokens.unsqueeze(0)
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
            output = ImageTokenOutput(
                global_features=global_features(tokens, mask),
                local_tokens=row_normalize(tokens),
                token_mask=mask,
                metadata={
                    "tap_path": "model.get_image_features(...).pooler_output after visual merger",
                    "token_mask_rule": "all merger-output tokens for the single image are valid",
                    "preprocessing": {
                        "resize_factor": self.factor,
                        "patch_size": self.patch_size,
                        "temporal_patch_size": self.temporal_patch_size,
                        "spatial_merge_size": self.merge_size,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                        "mean": self.processor.image_processor.image_mean,
                        "std": self.processor.image_processor.image_std,
                    },
                    "contains_text_tokens": False,
                },
            )
        return self._validate_output(output, require_grad=require_grad)

    def encode_image_layers(self, image: torch.Tensor, *, require_grad: bool) -> dict[str, ImageTokenOutput]:
        """Expose the validated pre-language interface as the available depth."""
        return {"interface": self.encode_image_tokens(image, require_grad=require_grad)}

    def encode_hierarchical_semantic(
        self, image: torch.Tensor, *, require_grad: bool
    ) -> tuple[dict[str, ImageTokenOutput], torch.Tensor]:
        output = self.encode_image_tokens(image, require_grad=require_grad)
        return {"interface": output}, output.global_features.float()

    def semantic_global(self, image: torch.Tensor, *, require_grad: bool) -> torch.Tensor:
        """Interface-token mean in Qwen's language-aligned hidden space."""
        _, semantic = self.encode_hierarchical_semantic(image, require_grad=require_grad)
        return semantic

    def encode_text_concepts(self, texts: list[str]) -> torch.Tensor:
        """Same-checkpoint lexical embeddings in the interface hidden space.

        Qwen's visual merger emits language-hidden-size vectors.  Averaging the
        frozen input-token embeddings supplies a text concept vector in that
        very same space without loading a second text encoder or using targets.
        """
        encoded = self.processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device).unsqueeze(-1)
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(input_ids)
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return F.normalize(pooled.float(), dim=-1)

    def prepare_answer_target(self, image: torch.Tensor, question: str, answer: str) -> QwenAnswerTarget:
        """Prepare a teacher-forced VQA answer target without entering the gradient path."""
        if image.shape[0] != 1:
            raise ValueError("Qwen answer supervision currently accepts one image.")
        height, width = image.shape[-2:]
        dummy_image = Image.new("RGB", (width, height))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": "Answer with a short answer only. Do not explain. " + question},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = self.processor(text=[text], images=[dummy_image], return_tensors="pt", padding=True)
        answer_ids = self.processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids = encoded["input_ids"]
        start = -1
        for index in range(input_ids.shape[1] - len(answer_ids), -1, -1):
            if input_ids[0, index:index + len(answer_ids)].tolist() == answer_ids:
                start = index
                break
        if start < 0:
            raise TapValidationError("Could not locate the canonical answer tokens in the Qwen chat template.")
        labels = torch.full_like(input_ids, -100)
        labels[:, start:start + len(answer_ids)] = input_ids[:, start:start + len(answer_ids)]
        return QwenAnswerTarget(
            input_ids=input_ids.to(self.device),
            attention_mask=encoded["attention_mask"].to(self.device),
            mm_token_type_ids=encoded["mm_token_type_ids"].to(self.device),
            labels=labels.to(self.device),
        )

    def answer_nll(self, image: torch.Tensor, target: QwenAnswerTarget) -> torch.Tensor:
        """Differentiable teacher-forced NLL for the specified VQA answer tokens."""
        pixel_values, grid = self._pixel_values(image)
        output = self.model(
            input_ids=target.input_ids,
            attention_mask=target.attention_mask,
            mm_token_type_ids=target.mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            labels=target.labels,
        )
        if output.loss is None or not torch.isfinite(output.loss):
            raise TapValidationError("Qwen answer NLL is not finite.")
        return output.loss.float()

    def unload(self) -> None:
        self.model.to("cpu")
        del self.model
