from __future__ import annotations

from dataclasses import dataclass, field
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
    question: str | None = None
    source_answer_text: str | None = None
    target_answer_text: str | None = None
    source_answer_keywords: list[str] = field(default_factory=list)
    target_answer_keywords: list[str] = field(default_factory=list)
    source_text_keywords: list[str] = field(default_factory=list)
    target_text_keywords: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


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
                question=item.get("question"),
                source_answer_text=item.get("source_answer_text"),
                target_answer_text=item.get("target_answer_text"),
                source_answer_keywords=normalize_keywords(
                    item["source_label"],
                    item.get("source_answer_keywords") or item.get("source_keywords"),
                ),
                target_answer_keywords=normalize_keywords(
                    item["target_label"],
                    item.get("target_answer_keywords") or item.get("target_keywords"),
                ),
                source_text_keywords=normalize_keywords(
                    item["source_label"],
                    item.get("source_text_keywords") or item.get("source_keywords"),
                ),
                target_text_keywords=normalize_keywords(
                    item["target_label"],
                    item.get("target_text_keywords") or item.get("target_keywords"),
                ),
                metadata=item.get("metadata", {}),
            )
        )
    return AttackManifest(
        dataset_name=payload.get("dataset_name", path.stem),
        metadata=payload.get("metadata", {}),
        items=items,
    )


def normalize_image_size(image_size: int | str | list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    if image_size is None:
        return None
    if isinstance(image_size, str):
        normalized = image_size.strip().lower()
        if normalized in {"original", "native", "keep"}:
            return None
        raise ValueError(f"Unsupported image_size string: {image_size!r}")
    if isinstance(image_size, int):
        return (image_size, image_size)
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        height, width = int(image_size[0]), int(image_size[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"image_size values must be positive, got {image_size!r}")
        return (height, width)
    raise ValueError(f"Unsupported image_size value: {image_size!r}")


def load_image_tensor(path: str | Path, image_size: int | str | list[int] | tuple[int, int] | None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(image)
    resize_to = normalize_image_size(image_size)
    if resize_to is None:
        return tensor
    return TF.resize(tensor, list(resize_to), antialias=True)


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(tensor.detach().cpu().clamp(0.0, 1.0))


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor_to_pil_image(tensor)
    image.save(path)
