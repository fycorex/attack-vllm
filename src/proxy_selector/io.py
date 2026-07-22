"""Atomic artifacts and PNG budget checks for the pilot."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from PIL import Image


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def save_png(image: torch.Tensor, path: Path) -> str:
    """Serialize one RGB tensor `[1,3,H,W]` losslessly and return its SHA-256."""
    if image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("save_png expects one RGB image [1,3,H,W].")
    pixels = image.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)[0]
    array = pixels.permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_png(path: Path, *, device: torch.device | str = "cpu") -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    pixels = torch.from_numpy(__import__("numpy").asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    return pixels.unsqueeze(0).to(device)


def png_linf(clean: torch.Tensor, path: Path) -> float:
    reloaded = load_png(path, device=clean.device)
    return float((reloaded - clean).abs().max().item())
