#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def trial_id(value: dict) -> str:
    geometry = f"__g{value.get('geometry_mode', 'none')}__gn{value.get('geometry_samples', 1)}"
    readable = f"{value['dataset']}__{value['surrogate_group']}__{value['mode']}__s{value['sigma']:.5f}__n{value['samples']}{geometry}__seed{value['seed']}"
    return readable.replace(".", "p") + "__" + hashlib.sha1(json.dumps(value, sort_keys=True).encode()).hexdigest()[:8]


def matrix(spec: dict, stage: str) -> list[dict]:
    stage_cfg = spec["stages"][stage]; noise = stage_cfg["noise"]
    values = []
    conditions = stage_cfg.get("explicit_conditions") or stage_cfg.get("promoted_candidates")
    if conditions:
        conditions = [{"mode": "none", "sigma": 0.0, "samples": 1}, *conditions] if stage_cfg.get("promoted_candidates") else conditions
        for dataset, group, seed, condition in itertools.product(stage_cfg["datasets"], stage_cfg["surrogate_groups"], stage_cfg["seeds"], conditions):
            values.append({"dataset": dataset, "surrogate_group": group, "mode": condition["mode"], "sigma": float(condition["sigma"]),
                           "samples": int(condition["samples"]), "seed": int(seed), "items": int(stage_cfg["items"]),
                           "geometry_mode": condition.get("geometry_mode", "none"),
                           "geometry_samples": int(condition.get("geometry_samples", 1)),
                           "steps": int(condition.get("steps", stage_cfg["steps"])), "stage": stage, "budget_label": condition.get("budget_label", "default")})
        return values
    for dataset, group, mode, sigma, samples, seed in itertools.product(
        stage_cfg["datasets"], stage_cfg["surrogate_groups"], noise["modes"], noise["sigmas"], noise["samples"], stage_cfg["seeds"]
    ):
        if mode in {"none", "paper_gaussian_single_sample"} and samples != 1: continue
        if mode == "none" and sigma != 0: continue
        if mode != "none" and sigma == 0: continue
        if mode == "antithetic_gaussian_eot" and samples % 2: continue
        values.append({"dataset": dataset, "surrogate_group": group, "mode": mode, "sigma": float(sigma), "samples": int(samples), "seed": int(seed),
                       "items": int(stage_cfg["items"]), "steps": int(stage_cfg["steps"]), "stage": stage})
    return values


def build_config(spec: dict, trial: dict, output: Path) -> tuple[Path, dict]:
    dataset = spec["datasets"][trial["dataset"]]; group = spec["surrogate_groups"][trial["surrogate_group"]]
    payload = yaml.safe_load(Path(dataset["attack_config"]).read_text())
    payload["experiment_name"] = f"noise-{trial_id(trial)}"; payload["paths"]["output_dir"] = str(output / "attack")
    payload["paths"]["manifest"] = dataset["manifest"]
    payload["runtime"].update({"seed": trial["seed"], "attack_limit": trial["items"], "sequential_surrogates": group.get("sequential", False)})
    payload["attack"].update({"steps": trial["steps"], "noise_mode": trial["mode"], "noise_sigma": trial["sigma"], "noise_samples": trial["samples"],
                              "geometry_mode": trial.get("geometry_mode", "none"),
                              "geometry_samples": trial.get("geometry_samples", 1),
                              "enable_gaussian": trial["mode"] in {"legacy", "paper_gaussian_single_sample"}})
    payload["surrogates"] = group["attack"]
    for key in ("caption_victim", "vqa_victim", "ocr_victim", "gpt_victim", "ollama_victim", "qwen_vl_victim"):
        if key in payload.get("evaluation", {}): payload["evaluation"][key]["enabled"] = False
    path = output / "effective_attack.yaml"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(yaml.safe_dump(payload, sort_keys=False))
    multiplier = int(payload["attack"].get("augmentation_batches", 1))
    noise_samples = trial["samples"] if trial["mode"].endswith("_eot") else 1
    geometry_samples = trial.get("geometry_samples", 1) if trial.get("geometry_mode", "none") not in {"legacy", "none"} else 1
    expected_forwards = trial["steps"] * len(group["attack"]) * multiplier * noise_samples * geometry_samples
    return path, {"dataset": dataset, "group": group, "expected_surrogate_forwards_per_item": expected_forwards}


def execute(spec: dict, trial: dict, root: Path, python: str) -> dict:
    directory = root / trial["stage"] / trial_id(trial); result_path = directory / "trial_result.json"
    if result_path.exists(): return json.loads(result_path.read_text())
    config, context = build_config(spec, trial, directory)
    attack_cmd = [python, "scripts/run_caption_attack.py", "--config", str(config)]
    if not (directory / "attack" / "summary.json").exists():
        with (directory / "attack.log").open("a") as log:
            subprocess.run(attack_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    heldout_path = directory / "heldout_summary.json"
    eval_cmd = [python, "scripts/evaluate_heldout_openclip.py", "--manifest", context["dataset"]["manifest"], "--attack-output", str(directory / "attack"),
                "--models", json.dumps(context["group"]["heldout"]), "--output", str(heldout_path)]
    if not heldout_path.exists():
        with (directory / "heldout.log").open("a") as log:
            subprocess.run(eval_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    heldout = json.loads(heldout_path.read_text()); result = {**trial, "trial_id": trial_id(trial),
                                                              "expected_surrogate_forwards_per_item": context["expected_surrogate_forwards_per_item"],
                                                              "heldout_macro_asr": heldout["heldout_macro_asr"],
                                                              "heldout_macro_margin_gain": heldout.get("heldout_macro_margin_gain", sum(r["margin_gain"] for r in heldout["items"]) / max(1, len(heldout["items"]))),
                                                              "status": "complete", "attack_command": attack_cmd, "heldout_command": eval_cmd}
    result_path.write_text(json.dumps(result, indent=2)); return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable multi-dataset noise transfer search.")
    parser.add_argument("--spec", required=True); parser.add_argument("--stage", required=True)
    parser.add_argument("--root", default="outputs/noise_search"); parser.add_argument("--plan", action="store_true")
    parser.add_argument("--max-trials", type=int); parser.add_argument("--python", default=sys.executable); parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mode"); parser.add_argument("--geometry-mode"); parser.add_argument("--dataset"); parser.add_argument("--surrogate-group")
    args = parser.parse_args(); spec = yaml.safe_load(Path(args.spec).read_text()); trials = matrix(spec, args.stage)
    if args.mode: trials = [t for t in trials if t["mode"] == args.mode]
    if args.geometry_mode: trials = [t for t in trials if t.get("geometry_mode", "none") == args.geometry_mode]
    if args.dataset: trials = [t for t in trials if t["dataset"] == args.dataset]
    if args.surrogate_group: trials = [t for t in trials if t["surrogate_group"] == args.surrogate_group]
    if args.max_trials is not None: trials = trials[:args.max_trials]
    if args.plan:
        print(json.dumps({"stage": args.stage, "trials": len(trials), "total_item_steps": sum(t["items"]*t["steps"] for t in trials), "matrix": trials}, indent=2)); return
    results = []
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // max(1, args.workers))))
    def guarded(trial):
        try: return execute(spec, trial, Path(args.root), args.python)
        except Exception as exc: return {**trial, "trial_id": trial_id(trial), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(guarded, trial) for trial in trials]
        for future in as_completed(futures): results.append(future.result())
    ranking = sorted((x for x in results if x["status"] == "complete"), key=lambda x: (x["heldout_macro_asr"], x.get("heldout_macro_margin_gain", 0)), reverse=True)
    output = Path(args.root) / args.stage / "ranking.json"; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(ranking, indent=2))
    print(json.dumps({"completed": len(ranking), "failed": len(results)-len(ranking), "best": ranking[:10]}, indent=2))


if __name__ == "__main__": main()
