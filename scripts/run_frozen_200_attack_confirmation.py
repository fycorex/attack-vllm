#!/usr/bin/env python3
"""Run an independent, frozen 200-image held-out transfer confirmation.

Candidate choice is read from an already finished search cycle. This runner
never reads API results and never tunes a candidate on its 200-image outcome.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import os
from typing import Any

import yaml

from experiment_validation import validate_attack_output
from surrogate_composition import build_attack_config, load_composition_spec
from transfer_eval import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/caption_attack_paper_eps8.yaml"
COMPOSITION = ROOT / "configs/research_cycle_cross_family.yaml"
FORWARD_BUDGET = 1200
EVAL_ROOT = ROOT.parent / "attack-vllm-eval-base"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen selection is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_or_screened(cycle: Path, final_name: str, screened_name: str) -> tuple[dict[str, Any], str]:
    """Prefer replicated final choices, but permit a time-bounded screened freeze.

    Both files are produced solely from held-out open-source measurements before
    this runner starts; the returned source is recorded for interpretation.
    """
    final_path = cycle / final_name
    if final_path.is_file():
        return load_json(final_path), "replicated_final"
    screened_path = cycle / screened_name
    if screened_path.is_file():
        return load_json(screened_path), "screen_only_due_to_time_limit"
    raise FileNotFoundError(f"Neither {final_path} nor {screened_path} exists.")


def choose_augmentation(value: dict[str, Any]) -> dict[str, Any] | None:
    rows = list(value.get("promoted", []))
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("selection_score", float("-inf"))))


def sample_multiplier(augmentation: dict[str, Any] | None) -> int:
    if not augmentation:
        return 1
    noise_samples = int(augmentation["samples"]) if str(augmentation["mode"]).endswith("_eot") else 1
    geometry_samples = int(augmentation.get("geometry_samples", 1)) if augmentation.get("geometry_mode", "none") != "none" else 1
    return noise_samples * geometry_samples


def conditions(selection: dict[str, Any], augmentation_selection: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = "single_openclip_vit"
    promoted = [name for name in selection.get("promoted", []) if name != baseline]
    if not promoted:
        raise RuntimeError("No non-baseline surrogate candidate was frozen by the search cycle.")
    best_surrogate = promoted[0]
    best_augmentation = choose_augmentation(augmentation_selection)
    result = [
        {"condition": "baseline", "surrogate_set": baseline, "augmentation": None},
        {"condition": "best_surrogate", "surrogate_set": best_surrogate, "augmentation": None},
    ]
    if best_augmentation is not None:
        result.extend([
            {"condition": "best_augmentation", "surrogate_set": baseline, "augmentation": best_augmentation},
            {"condition": "best_combined", "surrogate_set": best_surrogate, "augmentation": best_augmentation},
        ])
    return result


def forward_denominator(condition: dict[str, Any]) -> int:
    composition = load_composition_spec(COMPOSITION)
    model_count = len(composition.sets[condition["surrogate_set"]].models)
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    return model_count * int(base["attack"].get("augmentation_batches", 1)) * sample_multiplier(condition["augmentation"])


def matched_forward_budget(selected_conditions: list[dict[str, Any]]) -> int:
    """Return the smallest shared budget no smaller than the screening budget."""
    return math.lcm(FORWARD_BUDGET, *(forward_denominator(value) for value in selected_conditions))


def effective_config(
    condition: dict[str, Any], manifest: Path, directory: Path, items: int, seed: int, forward_budget: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    composition = load_composition_spec(COMPOSITION)
    model_count = len(composition.sets[condition["surrogate_set"]].models)
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    augmentation = condition["augmentation"]
    multiplier = sample_multiplier(augmentation)
    augmentation_batches = int(base["attack"].get("augmentation_batches", 1))
    denominator = model_count * augmentation_batches * multiplier
    if forward_budget % denominator:
        raise ValueError(f"Forward budget {forward_budget} is not divisible by condition denominator {denominator}")
    steps = forward_budget // denominator
    trial = {
        "trial_id": condition["condition"], "surrogate_set": condition["surrogate_set"],
        "seed": seed, "items": items, "steps": steps, "model_count": model_count,
        "augmentation_samples_per_step": augmentation_batches, "budget_mode": "equal_forwards",
        "forward_units_per_item": steps * model_count * augmentation_batches * multiplier,
    }
    experiment = {"datasets": {"caltech200": {"manifest": str(manifest)}}}
    config = build_attack_config(base, experiment, composition, {**trial, "dataset": "caltech200"}, directory, device="cuda")
    config["attack"].update({
        "noise_mode": "none", "noise_sigma": 0.0, "noise_samples": 1,
        "geometry_mode": "none", "geometry_samples": 1, "enable_gaussian": False,
    })
    if augmentation:
        config["attack"].update({
            "noise_mode": augmentation["mode"], "noise_sigma": float(augmentation["sigma"]),
            "noise_samples": int(augmentation["samples"]),
            "geometry_mode": augmentation.get("geometry_mode", "none"),
            "geometry_samples": int(augmentation.get("geometry_samples", 1)),
            "enable_gaussian": augmentation["mode"] in {"legacy", "paper_gaussian_single_sample"},
        })
    return config, {**trial, "augmentation": augmentation, "expected_forward_units_per_item": trial["forward_units_per_item"]}


def run_one(condition: dict[str, Any], root: Path, manifest: Path, items: int, seed: int, cache_dir: str, forward_budget: int) -> dict[str, Any]:
    directory = root / "frozen_200_attack" / condition["condition"]
    result_path = directory / "trial_result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    config, record = effective_config(condition, manifest, directory, items, seed, forward_budget)
    effective_path = directory / "effective_attack.yaml"
    effective_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    attack_command = [sys.executable, "scripts/run_caption_attack.py", "--config", str(effective_path)]
    with (directory / "attack.log").open("a", encoding="utf-8") as log:
        subprocess.run(attack_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    validation = validate_attack_output(
        directory / "attack", expected_items=items,
        epsilon=float(config["attack"]["epsilon"]),
        expected_forward_units_per_item=record["expected_forward_units_per_item"],
    )
    if not validation["valid"]:
        raise RuntimeError(f"Attack validation failed for {condition['condition']}: {validation}")
    heldout_path = directory / "heldout_summary.json"
    heldout_command = [
        sys.executable, "scripts/evaluate_surrogate_heldout.py", "--composition-config", str(COMPOSITION),
        "--manifest", str(manifest), "--attack-output", str(directory / "attack"),
        "--output", str(heldout_path), "--device", "cuda", "--cache-dir", cache_dir, "--limit", str(items),
    ]
    with (directory / "heldout.log").open("a", encoding="utf-8") as log:
        subprocess.run(heldout_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    result = {
        **record, "source_surrogate_set": record["surrogate_set"],
        "surrogate_set": condition["condition"],
        "stage": "frozen_200_attack", "dataset": "caption_caltech200",
        "condition": condition["condition"], "status": "complete",
        "heldout_macro_asr": heldout["heldout_macro_asr"],
        "heldout_macro_margin_gain": heldout["heldout_macro_margin_gain"],
        "attack_validation": validation, "attack_command": attack_command, "heldout_command": heldout_command,
    }
    atomic_write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen surrogate/noise transfer confirmation on 200 unique images.")
    parser.add_argument("--cycle-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-dir", default="models/open_clip")
    parser.add_argument("--run-api", action="store_true", help="Replay frozen 200-image candidates only after held-out confirmation.")
    parser.add_argument("--api-limit", type=int, default=50, help="Maximum paired items sent to the API per frozen candidate.")
    args = parser.parse_args()
    if args.items != 200:
        parser.error("This frozen confirmation is intentionally fixed at 200 unique images.")
    cycle = Path(args.cycle_root).resolve()
    manifest = Path(args.manifest).resolve()
    root = Path(args.output_root).resolve()
    selection, surrogate_selection_source = load_frozen_or_screened(
        cycle, "final_selection.json", "screen_selection.json"
    )
    augmentation_selection, augmentation_selection_source = load_frozen_or_screened(
        cycle, "augmentation_final_selection.json", "augmentation_screen_selection.json"
    )
    selected_conditions = conditions(selection, augmentation_selection)
    forward_budget = matched_forward_budget(selected_conditions)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "frozen_confirmation_spec.json", {
        "selection_source": "heldout_open_source_only", "api_results_used_for_selection": False,
        "surrogate_selection_maturity": surrogate_selection_source,
        "augmentation_selection_maturity": augmentation_selection_source,
        "cycle_root": str(cycle), "manifest": str(manifest), "items": args.items,
        "seed": args.seed, "forward_budget_per_item": forward_budget,
        "conditions": selected_conditions,
    })
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_one, value, root, manifest, args.items, args.seed, args.cache_dir, forward_budget): value
            for value in selected_conditions
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"condition": futures[future]["condition"], "error": f"{type(exc).__name__}: {exc}"})
    atomic_write_json(root / "confirmation_state.json", {
        "status": "complete" if not failures else "failed", "completed": len(results),
        "expected": len(selected_conditions), "failures": failures,
    })
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))
    subprocess.run([
        sys.executable, "scripts/analyze_surrogate_experiments.py", "--root", str(root),
        "--output", str(root / "analysis"), "--bootstrap-samples", "2000",
    ], cwd=ROOT, check=True)
    if not args.run_api:
        return
    eval_python = EVAL_ROOT / ".venv/bin/python"
    if not eval_python.is_file():
        eval_python = Path(sys.executable)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(EVAL_ROOT / "src"), str(EVAL_ROOT)])
    environment.setdefault("TECHUTOPIA_API_KEY", "research-cycle")
    analysis = root / "analysis" / "all_transfer_results.csv"
    frozen = root / "api" / "frozen_caption_caltech200.json"
    subprocess.run([
        str(eval_python), str(EVAL_ROOT / "scripts/freeze_api_candidates.py"),
        "--analysis-csv", str(analysis), "--stage", "frozen_200_attack",
        "--budget-mode", "equal_forwards", "--top-k", "3", "--minimum-seeds", "1",
        "--minimum-targets", "5", "--dataset", "caption_caltech200", "--baseline", "baseline",
        "--output", str(frozen),
    ], cwd=ROOT, env=environment, check=True)
    api_result = root / "api" / "results"
    subprocess.run([
        str(eval_python), str(EVAL_ROOT / "scripts/replay_multimodel_eval.py"),
        "--config", str(EVAL_ROOT / "configs/transferability_techutopia_available.yaml"),
        "--frozen-candidates", str(frozen), "--result-dir", str(api_result),
        "--allow-real-api", "--resume", "--limit", str(args.api_limit),
    ], cwd=ROOT, env=environment, check=True)
    subprocess.run([
        str(eval_python), str(EVAL_ROOT / "scripts/build_final_api_rubric.py"),
        "--result-dir", str(api_result), "--output", str(root / "api" / "rubric"),
    ], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
