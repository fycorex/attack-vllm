from __future__ import annotations

from copy import deepcopy

from .config import AttackConfig


def apply_ablation(config: AttackConfig, variant: str) -> AttackConfig:
    cfg = deepcopy(config)
    name = variant.strip().lower()
    if name == "full":
        return cfg
    if name == "no_drop_path":
        cfg.attack.enable_drop_path = False
        cfg.attack.drop_path_max_rate = 0.0
        return cfg
    if name == "no_patch_drop":
        cfg.attack.enable_patch_drop = False
        cfg.attack.patch_drop_rate = 0.0
        return cfg
    if name == "no_perturbation_average":
        cfg.attack.enable_perturbation_ema = False
        cfg.attack.perturbation_ema_decay = 0.0
        return cfg
    if name == "no_gaussian":
        cfg.attack.enable_gaussian = False
        cfg.attack.gaussian_prob = 0.0
        return cfg
    if name == "no_crop_pad_resize_jpeg":
        cfg.attack.enable_crop = False
        cfg.attack.crop_prob = 0.0
        cfg.attack.enable_pad = False
        cfg.attack.pad_prob = 0.0
        cfg.attack.enable_jpeg = False
        cfg.attack.jpeg_prob = 0.0
        return cfg
    if name == "proxy_only":
        cfg.evaluation.caption_victim.enabled = False
        return cfg
    raise ValueError(f"Unknown ablation variant: {variant}")


def list_default_ablations() -> list[str]:
    return [
        "full",
        "no_drop_path",
        "no_patch_drop",
        "no_perturbation_average",
        "no_gaussian",
        "no_crop_pad_resize_jpeg",
        "proxy_only",
    ]
