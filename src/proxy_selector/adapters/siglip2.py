"""Exact Hugging Face SigLIP2 So400m image-token adapter."""

from __future__ import annotations

from pathlib import Path

from .hf_vision import HFVisionTokenAdapter


class SigLIP2Adapter(HFVisionTokenAdapter):
    def __init__(self, checkpoint_path: Path, device: str = "cuda") -> None:
        super().__init__(
            "google/siglip2-so400m-patch14-384",
            checkpoint_path,
            family="SigLIP2",
            drop_first_token=False,
            device=device,
        )
