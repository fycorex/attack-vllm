from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True)
class ModelMetadata:
    model_name: str
    pretrained: str
    input_size: int
    architecture_family: str
    objective_family: str
    pretraining_family: str
    patch_size: int | None = None
    backend: str = "open_clip"
    evaluation_role: str = "primary"
    rationale: str = ""

    def victim_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "backend": self.backend,
            "input_size": self.input_size,
            "patch_size": self.patch_size,
        }


@dataclass(frozen=True)
class SurrogateSet:
    name: str
    models: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class CompositionSpec:
    models: dict[str, ModelMetadata]
    sets: dict[str, SurrogateSet]
    heldout_models: tuple[str, ...]

    def resolve(self, name: str) -> list[ModelMetadata]:
        if name not in self.sets:
            raise KeyError(f"Unknown surrogate set: {name}")
        return [self.models[model_id] for model_id in self.sets[name].models]

    def validate(self) -> None:
        unknown_heldout = set(self.heldout_models) - set(self.models)
        if unknown_heldout:
            raise ValueError(f"Unknown held-out models: {sorted(unknown_heldout)}")
        for name, value in self.sets.items():
            if not value.models:
                raise ValueError(f"Surrogate set {name} is empty")
            if len(set(value.models)) != len(value.models):
                raise ValueError(f"Surrogate set {name} contains duplicate models")
            unknown = set(value.models) - set(self.models)
            if unknown:
                raise ValueError(f"Surrogate set {name} references unknown models: {sorted(unknown)}")
            overlap = set(value.models).intersection(self.heldout_models)
            if overlap:
                raise ValueError(f"Surrogate set {name} overlaps held-out models: {sorted(overlap)}")


def load_composition_spec(path: str | Path) -> CompositionSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    required = {"model_name", "pretrained", "input_size", "architecture_family", "objective_family", "pretraining_family"}
    models = {}
    for model_id, value in raw.get("models", {}).items():
        missing = required - set(value)
        if missing:
            raise ValueError(f"Model {model_id} lacks explicit metadata: {sorted(missing)}")
        models[model_id] = ModelMetadata(**value)
    sets = {
        name: SurrogateSet(name=name, models=tuple(value["models"]), rationale=str(value.get("rationale", "")))
        for name, value in raw.get("sets", {}).items()
    }
    spec = CompositionSpec(models=models, sets=sets, heldout_models=tuple(raw.get("heldout_models", [])))
    spec.validate()
    return spec


def gradient_diagnostics(gradients: dict[str, torch.Tensor], eps: float = 1e-12) -> dict[str, Any]:
    names = list(gradients)
    if not names:
        return {"models": {}, "pairwise_cosine": {}}
    flattened = {name: gradients[name].detach().float().flatten() for name in names}
    norms = {name: float(torch.linalg.vector_norm(value).item()) for name, value in flattened.items()}
    pairwise = {}
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            denominator = norms[first] * norms[second]
            cosine = float(torch.dot(flattened[first], flattened[second]).item() / denominator) if denominator > eps else 0.0
            pairwise[f"{first}__{second}"] = cosine
    weight = 1.0 / len(names)
    return {
        "models": {name: {"gradient_norm": norms[name], "effective_equal_weight": weight} for name in names},
        "pairwise_cosine": pairwise,
    }


def stage_trials(experiment: dict, composition: CompositionSpec, stage: str) -> list[dict]:
    stage_config = experiment["stages"][stage]
    trials = []
    for dataset, set_name, seed in itertools.product(stage_config["datasets"], stage_config["sets"], stage_config["seeds"]):
        model_count = len(composition.sets[set_name].models)
        augmentation_samples = int(experiment["datasets"][dataset].get("augmentation_samples_per_step", 1))
        if stage_config["budget_mode"] == "equal_forwards":
            steps = max(1, int(stage_config["reference_steps"]) // model_count)
        else:
            steps = int(stage_config["steps"])
        trial = {"stage": stage, "dataset": dataset, "surrogate_set": set_name, "seed": int(seed),
                 "items": int(stage_config["items"]), "steps": steps, "model_count": model_count,
                 "augmentation_samples_per_step": augmentation_samples,
                 "budget_mode": stage_config["budget_mode"],
                 "forward_units_per_item": steps * model_count * augmentation_samples}
        trial["trial_id"] = hashlib.sha256(json.dumps(trial, sort_keys=True).encode()).hexdigest()[:12]
        trials.append(trial)
    return trials


def build_attack_config(base: dict, experiment: dict, composition: CompositionSpec, trial: dict, output: Path, device: str | None = None) -> dict:
    value = json.loads(json.dumps(base))
    dataset = experiment["datasets"][trial["dataset"]]
    value["experiment_name"] = f"surrogate-{trial['trial_id']}"
    value["paths"]["manifest"] = dataset["manifest"]
    value["paths"]["output_dir"] = str(output / "attack")
    value["runtime"].update({"seed": trial["seed"], "attack_limit": trial["items"], "attack_offset": 0,
                             "sequential_surrogates": False,
                             "attack_batch_size": int(dataset.get("attack_batch_size", 1))})
    if device is not None:
        value["runtime"]["device"] = device
    value["attack"].update({"steps": trial["steps"], "metrics_interval": max(1, trial["steps"] // 10)})
    value["surrogates"] = [composition.models[model_id].victim_config() | {"enabled": True} for model_id in composition.sets[trial["surrogate_set"]].models]
    evaluation = value.setdefault("evaluation", {})
    for victim in ("caption_victim", "vqa_victim", "ocr_victim", "gpt_victim", "ollama_victim", "qwen_vl_victim"):
        evaluation.setdefault(victim, {})["enabled"] = False
    return value
