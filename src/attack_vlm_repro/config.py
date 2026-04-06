from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import yaml


@dataclass
class SurrogateConfig:
    model_name: str
    pretrained: str
    input_size: int = 224
    enabled: bool = True
    patch_size: int | None = None


@dataclass
class AttackHyperParams:
    epsilon: float = 16.0 / 255.0
    step_size: float = 1.0 / 255.0
    steps: int = 20
    temperature: float = 0.1
    top_k: int = 4
    patch_drop_rate: float = 0.20
    drop_path_max_rate: float = 0.10
    perturbation_ema_decay: float = 0.99
    gaussian_prob: float = 0.5
    gaussian_scale_multiplier: float = 0.25
    crop_prob: float = 0.5
    crop_scale_min: float = 0.80
    crop_scale_max: float = 1.00
    crop_ratio_min: float = 0.75
    crop_ratio_max: float = 1.3333333333
    pad_prob: float = 0.5
    jpeg_prob: float = 0.5
    jpeg_quality_min: float = 0.5
    jpeg_quality_max: float = 1.0
    enable_patch_drop: bool = True
    enable_drop_path: bool = True
    enable_perturbation_ema: bool = True
    enable_gaussian: bool = True
    enable_crop: bool = True
    enable_pad: bool = True
    enable_jpeg: bool = True


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    seed: int = 123
    attack_limit: int | None = 4
    sequential_surrogates: bool = True


@dataclass
class CaptionVictimConfig:
    enabled: bool = False
    model_name: str = "Salesforce/blip-image-captioning-base"
    device: str = "cuda"
    use_fp16: bool = True
    sequential_loading: bool = True
    max_new_tokens: int = 24
    num_beams: int = 3
    prompt: str | None = None
    require_source_absent: bool = True


@dataclass
class VQAVictimConfig:
    enabled: bool = False
    model_name: str = "Salesforce/blip-vqa-base"
    device: str = "cuda"
    use_fp16: bool = True
    sequential_loading: bool = True
    max_new_tokens: int = 10
    num_beams: int = 3
    question_fallback: str = "What is the main object in the image?"
    require_source_absent: bool = True


@dataclass
class EvaluationConfig:
    success_margin_threshold: float = 0.0
    caption_victim: CaptionVictimConfig = field(default_factory=CaptionVictimConfig)
    vqa_victim: VQAVictimConfig = field(default_factory=VQAVictimConfig)


@dataclass
class PathsConfig:
    manifest: str = "data/cifar10_caption_attack_demo/manifest.json"
    output_dir: str = "outputs/caption_attack_demo"
    model_cache_dir: str = "models"


@dataclass
class AttackConfig:
    experiment_name: str = "caption-attack-demo"
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    attack: AttackHyperParams = field(default_factory=AttackHyperParams)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    surrogates: list[SurrogateConfig] = field(default_factory=list)


def load_config(path: str | Path) -> AttackConfig:
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    payload = json.loads(raw_text) if path.suffix.lower() == ".json" else yaml.safe_load(raw_text)

    evaluation_payload = payload.get("evaluation", {})
    caption_victim_payload = evaluation_payload.get("caption_victim", {})
    vqa_victim_payload = evaluation_payload.get("vqa_victim", {})

    cfg = AttackConfig(
        experiment_name=payload.get("experiment_name", "caption-attack-demo"),
        paths=PathsConfig(**payload.get("paths", {})),
        runtime=RuntimeConfig(**payload.get("runtime", {})),
        attack=AttackHyperParams(**payload.get("attack", {})),
        evaluation=EvaluationConfig(
            success_margin_threshold=evaluation_payload.get("success_margin_threshold", 0.0),
            caption_victim=CaptionVictimConfig(**caption_victim_payload),
            vqa_victim=VQAVictimConfig(**vqa_victim_payload),
        ),
        surrogates=[SurrogateConfig(**item) for item in payload.get("surrogates", [])],
    )
    if not cfg.surrogates:
        cfg.surrogates = [
            SurrogateConfig(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", input_size=224),
            SurrogateConfig(model_name="ViT-B-16", pretrained="laion2b_s34b_b88k", input_size=224),
        ]
    return cfg
