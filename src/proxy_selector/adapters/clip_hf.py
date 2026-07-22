"""Exact Hugging Face CLIP ViT-L/14 image-token adapter."""

from __future__ import annotations

from pathlib import Path

from .hf_vision import HFVisionTokenAdapter


class CLIPHFAdapter(HFVisionTokenAdapter):
    def __init__(self, checkpoint_path: Path, device: str = "cuda") -> None:
        super().__init__(
            "openai/clip-vit-large-patch14",
            checkpoint_path,
            family="CLIP",
            drop_first_token=True,
            device=device,
        )
