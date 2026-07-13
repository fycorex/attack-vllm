from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms import functional as TF
from PIL import Image


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {path}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def validate_attack_output(attack_output: str | Path, *, expected_items: int, epsilon: float,
                           expected_forward_units_per_item: int, quantization_tolerance: float = 1.0 / 255.0 + 1e-6) -> dict[str, Any]:
    root = Path(attack_output)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"Missing attack summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    items = summary.get("items", [])
    if len(items) != expected_items:
        raise ValueError(f"Expected {expected_items} items, found {len(items)}")
    maximum_linf = 0.0
    for item in items:
        item_id = item["item_id"]
        item_dir = root / item_id
        clean_path, adversarial_path = item_dir / "clean.png", item_dir / "adversarial.png"
        metrics_path = item_dir / "metrics.json"
        for required in (clean_path, adversarial_path, metrics_path):
            if not required.is_file():
                raise ValueError(f"Missing required output: {required}")
        clean = TF.to_tensor(Image.open(clean_path).convert("RGB"))
        adversarial = TF.to_tensor(Image.open(adversarial_path).convert("RGB"))
        if clean.shape != adversarial.shape:
            raise ValueError(f"Image shape mismatch for {item_id}")
        linf = float((adversarial - clean).abs().max())
        maximum_linf = max(maximum_linf, linf)
        if linf > epsilon + quantization_tolerance:
            raise ValueError(f"L-infinity violation for {item_id}: {linf} > {epsilon}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        _assert_finite(metrics, item_id)
        forwards = metrics.get("composition_diagnostics", {}).get("total_surrogate_forwards")
        if forwards != expected_forward_units_per_item:
            raise ValueError(f"Forward accounting mismatch for {item_id}: {forwards} != {expected_forward_units_per_item}")
    return {"valid": True, "items": len(items), "maximum_linf": maximum_linf,
            "epsilon": epsilon, "expected_forward_units_per_item": expected_forward_units_per_item}
