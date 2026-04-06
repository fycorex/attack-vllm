from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import torch
from PIL import Image
from torchvision.transforms import functional as TF


@dataclass
class AttackItem:
    item_id: str
    image_path: Path
    source_label: str
    target_label: str
    positive_image_paths: list[Path]
    negative_image_paths: list[Path]
    source_keywords: list[str]
    target_keywords: list[str]


@dataclass
class AttackManifest:
    dataset_name: str
    metadata: dict
    items: list[AttackItem]


def normalize_keywords(label: str, keywords: list[str] | None = None) -> list[str]:
    values = [label]
    if keywords:
        values.extend(keywords)
    normalized: list[str] = []
    seen = set()
    for value in values:
        candidate = re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))
        if not candidate or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    return normalized


def load_manifest(path: str | Path) -> AttackManifest:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    items: list[AttackItem] = []
    for item in payload["items"]:
        items.append(
            AttackItem(
                item_id=item["id"],
                image_path=Path(item["image_path"]),
                source_label=item["source_label"],
                target_label=item["target_label"],
                positive_image_paths=[Path(p) for p in item["positive_image_paths"]],
                negative_image_paths=[Path(p) for p in item["negative_image_paths"]],
                source_keywords=normalize_keywords(item["source_label"], item.get("source_keywords")),
                target_keywords=normalize_keywords(item["target_label"], item.get("target_keywords")),
            )
        )
    return AttackManifest(
        dataset_name=payload.get("dataset_name", path.stem),
        metadata=payload.get("metadata", {}),
        items=items,
    )


def load_image_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(image)
    return TF.resize(tensor, [image_size, image_size], antialias=True)


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(tensor.detach().cpu().clamp(0.0, 1.0))


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor_to_pil_image(tensor)
    image.save(path)
