from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import yaml


SUPPORTED_PROFILES = ("light", "heavy", "api")


@dataclass
class SurrogateConfig:
    model_name: str
    pretrained: str
    input_size: int = 224
    enabled: bool = True
    patch_size: int | None = None
    use_fp16: bool = True


@dataclass
class AttackHyperParams:
    epsilon: float = 16.0 / 255.0
    step_size: float = 1.0 / 255.0
    steps: int = 20
    augmentation_batches: int = 1
    temperature: float = 0.1
    top_k: int = 4
    relative_proxy_weight: float = 0.0
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
    attack_offset: int = 0
    sequential_surrogates: bool = True
    enable_tf32: bool = True
    cudnn_benchmark: bool = True


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
class OCRVictimConfig:
    enabled: bool = False
    backend: str = "tesseract"
    model_name: str = "microsoft/trocr-small-printed"
    device: str = "cuda"
    use_fp16: bool = True
    sequential_loading: bool = True
    max_new_tokens: int = 16
    num_beams: int = 1
    tesseract_psm: int = 7
    require_source_absent: bool = True


@dataclass
class GPTVictimConfig:
    enabled: bool = False
    model_name: str = "gpt-4o"
    task_type: str = "vqa"
    api_mode: str = "auto"
    prompt_mode: str = "freeform"
    success_mode: str = "keyword"
    question_fallback: str = "What is the main object in the image?"
    caption_prompt: str = "Provide a concise description of the image using no more than three sentences."
    require_source_absent: bool = True
    max_output_tokens: int = 64
    temperature: float = 0.0
    reasoning_effort: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    request_timeout_seconds: int = 90
    max_retries: int = 4
    retry_backoff_seconds: float = 10.0
    request_pause_seconds: float = 0.0
    judge_model_name: str = "gpt-4o"
    judge_api_mode: str = "auto"
    judge_max_output_tokens: int = 128
    judge_temperature: float = 0.0
    judge_reasoning_effort: str | None = None
    judge_api_key_env: str = "OPENAI_API_KEY"
    judge_base_url: str = "https://api.openai.com/v1"


@dataclass
class OllamaVictimConfig:
    """Configuration for local ollama VLM victim (e.g., QwenVL)."""
    enabled: bool = False
    model_name: str = "qwen3-vl:4b"
    base_url: str = "http://localhost:11434"
    api_endpoint: str = "api/generate"
    task_type: str = "vqa"
    max_tokens: int = 128
    temperature: float = 0.0
    question_fallback: str = "What is the main object in the image?"
    require_source_absent: bool = True
    health_check_timeout: float = 5.0
    request_timeout: float = 60.0
    max_retries: int = 3
    retry_backoff: float = 2.0


@dataclass
class HuggingFaceQwenVLConfig:
    """Configuration for HuggingFace QwenVL victim as fallback."""
    enabled: bool = False
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "cuda"
    use_fp16: bool = True
    max_new_tokens: int = 64
    sequential_loading: bool = True
    question_fallback: str = "What is the main object in the image?"
    require_source_absent: bool = True


@dataclass
class EvaluationConfig:
    success_margin_threshold: float = 0.0
    caption_victim: CaptionVictimConfig = field(default_factory=CaptionVictimConfig)
    vqa_victim: VQAVictimConfig = field(default_factory=VQAVictimConfig)
    ocr_victim: OCRVictimConfig = field(default_factory=OCRVictimConfig)
    gpt_victim: GPTVictimConfig = field(default_factory=GPTVictimConfig)
    ollama_victim: OllamaVictimConfig = field(default_factory=OllamaVictimConfig)
    qwen_vl_victim: HuggingFaceQwenVLConfig = field(default_factory=HuggingFaceQwenVLConfig)


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
    run_profile: str = "custom"
    profile_metadata: dict = field(default_factory=dict)


def config_to_dict(config: AttackConfig) -> dict:
    return asdict(config)


def enabled_surrogate_names(config: AttackConfig) -> list[str]:
    return [f"{item.model_name}:{item.pretrained}" for item in config.surrogates if item.enabled]


def _clip_first_heavy_surrogates() -> list[SurrogateConfig]:
    return [
        SurrogateConfig(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", input_size=224),
        SurrogateConfig(model_name="ViT-B-16", pretrained="laion2b_s34b_b88k", input_size=224),
        SurrogateConfig(model_name="RN50", pretrained="openai", input_size=224),
        SurrogateConfig(model_name="RN101", pretrained="openai", input_size=224),
        SurrogateConfig(model_name="ViT-L-14", pretrained="openai", input_size=224),
    ]


def apply_profile(config: AttackConfig, profile: str | None) -> None:
    if profile is None:
        config.run_profile = "custom"
        config.profile_metadata = {
            "profile": "custom",
            "description": "Config-only run; no runtime profile overlay applied.",
            "enabled_surrogates": enabled_surrogate_names(config),
        }
        return

    normalized = profile.strip().lower()
    if normalized not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}. Expected one of: {', '.join(SUPPORTED_PROFILES)}")

    config.run_profile = normalized
    applied_overrides: dict[str, object] = {}
    if normalized == "light":
        if config.runtime.attack_limit is None:
            config.runtime.attack_limit = 4
        else:
            config.runtime.attack_limit = min(config.runtime.attack_limit, 4)
        config.attack.steps = min(config.attack.steps, 50)
        config.attack.augmentation_batches = 1
        for idx, surrogate in enumerate(config.surrogates):
            surrogate.enabled = idx < 2
        config.runtime.sequential_surrogates = True
        applied_overrides = {
            "runtime.attack_limit": config.runtime.attack_limit,
            "attack.steps_max": 50,
            "attack.augmentation_batches": 1,
            "surrogate_policy": "first_two_enabled",
            "runtime.sequential_surrogates": True,
        }
    elif normalized == "heavy":
        config.surrogates = _clip_first_heavy_surrogates()
        if config.runtime.attack_limit is None:
            config.runtime.attack_limit = 8
        else:
            config.runtime.attack_limit = max(config.runtime.attack_limit, 8)
        config.attack.steps = max(config.attack.steps, 120)
        config.attack.augmentation_batches = max(config.attack.augmentation_batches, 4)
        config.attack.top_k = max(config.attack.top_k, 4)
        config.runtime.enable_tf32 = True
        config.runtime.cudnn_benchmark = True
        config.runtime.sequential_surrogates = False
        applied_overrides = {
            "surrogate_policy": "clip_first_a6000_heavy",
            "runtime.attack_limit_min": 8,
            "attack.steps_min": 120,
            "attack.augmentation_batches_min": 4,
            "attack.top_k_min": 4,
            "runtime.enable_tf32": True,
            "runtime.cudnn_benchmark": True,
            "runtime.sequential_surrogates": False,
        }
    elif normalized == "api":
        config.evaluation.gpt_victim.enabled = True
        config.evaluation.gpt_victim.api_mode = config.evaluation.gpt_victim.api_mode or "auto"
        config.evaluation.gpt_victim.success_mode = config.evaluation.gpt_victim.success_mode or "judge"
        config.runtime.sequential_surrogates = True
        applied_overrides = {
            "evaluation.gpt_victim.enabled": True,
            "runtime.sequential_surrogates": True,
            "workflow_note": "API profile enables live GPT victim evaluation for this runner; replay remains a separate companion script.",
        }

    config.profile_metadata = {
        "profile": normalized,
        "applied_overrides": applied_overrides,
        "enabled_surrogates": enabled_surrogate_names(config),
    }


def load_config(path: str | Path) -> AttackConfig:
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    payload = json.loads(raw_text) if path.suffix.lower() == ".json" else yaml.safe_load(raw_text)

    evaluation_payload = payload.get("evaluation", {})
    caption_victim_payload = evaluation_payload.get("caption_victim", {})
    vqa_victim_payload = evaluation_payload.get("vqa_victim", {})
    ocr_victim_payload = evaluation_payload.get("ocr_victim", {})
    gpt_victim_payload = evaluation_payload.get("gpt_victim", {})
    ollama_victim_payload = evaluation_payload.get("ollama_victim", {})
    qwen_vl_victim_payload = evaluation_payload.get("qwen_vl_victim", {})

    cfg = AttackConfig(
        experiment_name=payload.get("experiment_name", "caption-attack-demo"),
        paths=PathsConfig(**payload.get("paths", {})),
        runtime=RuntimeConfig(**payload.get("runtime", {})),
        attack=AttackHyperParams(**payload.get("attack", {})),
        evaluation=EvaluationConfig(
            success_margin_threshold=evaluation_payload.get("success_margin_threshold", 0.0),
            caption_victim=CaptionVictimConfig(**caption_victim_payload),
            vqa_victim=VQAVictimConfig(**vqa_victim_payload),
            ocr_victim=OCRVictimConfig(**ocr_victim_payload),
            gpt_victim=GPTVictimConfig(**gpt_victim_payload),
            ollama_victim=OllamaVictimConfig(**ollama_victim_payload),
            qwen_vl_victim=HuggingFaceQwenVLConfig(**qwen_vl_victim_payload),
        ),
        surrogates=[SurrogateConfig(**item) for item in payload.get("surrogates", [])],
    )
    if not cfg.surrogates:
        cfg.surrogates = [
            SurrogateConfig(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", input_size=224),
            SurrogateConfig(model_name="ViT-B-16", pretrained="laion2b_s34b_b88k", input_size=224),
        ]
    apply_profile(cfg, None)
    return cfg
