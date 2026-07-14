#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import torch
import yaml

from scripts.run_ensemble_gated_pipeline import completed_trials, stage_specs, validate_stage
from scripts.select_transfer_candidates import promote, score_candidates
from transfer_eval import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT.parent / "attack-vllm-eval-base"
GEOMETRIC_ROOT = ROOT.parent / "attack-vllm-geometric"
ALIGNMENT_ROOT = ROOT.parent / "attack-vllm-alignment"
DEFAULT_SPEC = ROOT / "configs/research_cycle_cross_family_experiments.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/transfer_search_cycle"
API_CONFIGS = {
    "caption_caltech": EVAL_ROOT / "configs/transferability_techutopia_available.yaml",
    "llava_vqa": EVAL_ROOT / "configs/transferability_techutopia_vqa.yaml",
    "receipt_ocr": EVAL_ROOT / "configs/transferability_techutopia_receipt.yaml",
}


def command_string(command: list[str]) -> str:
    return " ".join(command)


def write_state(path: Path, **values: Any) -> None:
    atomic_write_json(path, {"updated_at": datetime.now(timezone.utc).isoformat(), **values})


def expected_trials(spec: dict[str, Any], stage: str) -> int:
    value = spec["stages"][stage]
    return len(value["datasets"]) * len(value["sets"]) * len(value["seeds"])


def run_bounded_stage(
    spec: dict[str, Any],
    spec_path: Path,
    stage: str,
    root: Path,
    jobs: int,
    cache_dir: Path,
    state_path: Path,
    deadline: float,
) -> None:
    generated = root / "generated_specs" / stage
    generated.mkdir(parents=True, exist_ok=True)
    paths = stage_specs(spec, stage, generated)
    running: list[tuple[Path, subprocess.Popen, Any]] = []
    queue = list(paths)
    log_dir = root / "runner_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    while queue or running:
        while queue and len(running) < jobs and time.monotonic() < deadline:
            path = queue.pop(0)
            log = (log_dir / f"{path.stem}.log").open("a", encoding="utf-8")
            command = [
                sys.executable, "scripts/run_surrogate_experiments.py",
                "--spec", str(path), "--stage", stage, "--root", str(root),
                "--device", "cuda", "--cache-dir", str(cache_dir),
            ]
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((path, process, log))
        remaining = []
        for path, process, log in running:
            code = process.poll()
            if code is None:
                remaining.append((path, process, log))
            else:
                log.close()
                if code:
                    failures.append((str(path), code))
        running = remaining
        write_state(
            state_path, status=f"running_{stage}",
            completed=len(completed_trials(root, stage)), expected=expected_trials(spec, stage),
            queued=len(queue), active=len(running), failures=failures,
        )
        if failures:
            raise RuntimeError(f"{stage} worker failures: {failures}")
        if time.monotonic() >= deadline and not running:
            break
        time.sleep(10)
    if queue:
        raise TimeoutError(f"Soft deadline reached with {len(queue)} {stage} set runners not started")
    validate_stage(root, stage, expected_trials(spec, stage))


def run_analysis(root: Path, output: Path) -> Path:
    command = [
        sys.executable, "scripts/analyze_surrogate_experiments.py",
        "--root", str(root), "--output", str(output), "--bootstrap-samples", "2000",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return output / "all_transfer_results.csv"


def select_and_write(
    analysis_csv: Path,
    composition: Path,
    stage: str,
    output: Path,
    count: int,
) -> dict[str, Any]:
    with analysis_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ranking = score_candidates(rows, composition, stage, "equal_forwards")
    result = {
        "schema_version": 1,
        "selection_source": "heldout_open_source_only",
        "api_results_used_for_selection": False,
        "stage": stage,
        "promoted": promote(ranking, count),
        "ranking": ranking,
    }
    atomic_write_json(output, result)
    return result


def write_replication_spec(
    base: dict[str, Any], selected: list[str], output: Path,
) -> dict[str, Any]:
    value = copy.deepcopy(base)
    baseline = "single_openclip_vit"
    sets = [baseline, *[name for name in selected if name != baseline]]
    value["stages"]["cross_family_replication"]["sets"] = sets
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return value


def run_measurements(spec: dict[str, Any], output: Path, cache_dir: Path) -> None:
    composition = ROOT / spec["composition_config"]
    set_names = [name for name in spec["stages"]["cross_family_screen"]["sets"] if name.startswith("single_")]
    limits = {"caption_caltech": 200, "llava_vqa": 20, "receipt_ocr": 19}
    for dataset, limit in limits.items():
        destination = output / "alignment" / dataset
        if (destination / "run_manifest.json").is_file():
            continue
        command = [
            sys.executable, "scripts/measure_surrogate_geometry.py",
            "--config", str(composition), "--manifest", spec["datasets"][dataset]["manifest"],
            "--sets", *set_names, "--limit", str(limit), "--deduplicate-images",
            "--output", str(destination), "--device", "cuda", "--batch-size", "8",
            "--cache-dir", str(cache_dir), "--subsample-sizes", "10", "20", "50", "100",
            "--subsample-repeats", "100",
        ]
        subprocess.run(command, cwd=ROOT, check=True)


def start_augmentation_stage(
    spec_path: Path,
    stage: str,
    output: Path,
    workers: int,
) -> tuple[subprocess.Popen, Any]:
    python = GEOMETRIC_ROOT / ".venv/bin/python"
    log_path = output / "runner_logs" / f"augmentation_{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    command = [
        str(python), "scripts/run_noise_search.py", "--spec", str(spec_path),
        "--stage", stage, "--root", str(output / "augmentation"),
        "--workers", str(workers), "--python", str(python),
    ]
    # The parent cycle runs with this worktree's src first on PYTHONPATH.  The
    # geometric runner has a different AttackHyperParams schema, so inheriting
    # that path silently imports the wrong config module and rejects its noise
    # fields.  Pin the child to its own source tree.
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(GEOMETRIC_ROOT / "src"), str(GEOMETRIC_ROOT)])
    process = subprocess.Popen(command, cwd=GEOMETRIC_ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment)
    return process, log


def wait_process(process: subprocess.Popen, log: Any, label: str) -> None:
    code = process.wait()
    log.close()
    if code:
        raise RuntimeError(f"{label} failed with exit code {code}")


def _augmentation_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["mode"], float(row["sigma"]), int(row["samples"]),
        row.get("geometry_mode", "none"), int(row.get("geometry_samples", 1)),
        int(row["steps"]), row.get("budget_label", "default"),
    )


def select_augmentations(root: Path, stage: str, output: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for path in sorted((root / stage).glob("*/trial_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "complete":
            grouped.setdefault(_augmentation_key(row), []).append(row)
    scored = []
    for key, rows in grouped.items():
        task_scores: dict[str, list[float]] = {}
        for row in rows:
            task_scores.setdefault(row["dataset"], []).append(float(row["heldout_macro_asr"]))
        task_means = {name: statistics.fmean(values) for name, values in sorted(task_scores.items())}
        macro = statistics.fmean(task_means.values()) if task_means else 0.0
        worst = min(task_means.values(), default=0.0)
        scored.append({
            "mode": key[0], "sigma": key[1], "samples": key[2],
            "geometry_mode": key[3], "geometry_samples": key[4], "steps": key[5],
            "budget_label": key[6], "task_asr": task_means,
            "macro_asr": macro, "worst_task_asr": worst,
            "selection_score": 0.7 * macro + 0.3 * worst,
            "complete_datasets": len(task_means),
        })
    eligible = [row for row in scored if row["complete_datasets"] == 3]
    eligible.sort(key=lambda row: row["selection_score"], reverse=True)
    categories = [
        lambda row: row["mode"] != "none" and row["geometry_mode"] == "none",
        lambda row: row["mode"] == "none" and row["geometry_mode"] != "none",
        lambda row: row["mode"] != "none" and row["geometry_mode"] != "none",
    ]
    selected = []
    for predicate in categories:
        value = next((row for row in eligible if predicate(row)), None)
        if value is not None and _augmentation_key(value) not in {_augmentation_key(row) for row in selected}:
            selected.append(value)
    result = {
        "schema_version": 1,
        "selection_source": "heldout_open_source_only",
        "api_results_used_for_selection": False,
        "stage": stage,
        "promoted": selected,
        "ranking": eligible,
    }
    atomic_write_json(output, result)
    return selected


def write_augmentation_replication_spec(
    source: Path,
    selected: list[dict[str, Any]],
    output: Path,
) -> str:
    spec = yaml.safe_load(source.read_text(encoding="utf-8"))
    stage = "strict_equal_forward_replication"
    spec["stages"] = {stage: {
        "datasets": ["caption_caltech", "llava_vqa", "receipt_ocr"],
        "surrogate_groups": ["vit_resnet_diverse"],
        "items": 19,
        "steps": 200,
        "seeds": [123, 2026],
        "noise": {"modes": [], "sigmas": [], "samples": []},
        "promoted_candidates": [{
            key: row[key] for key in (
                "mode", "sigma", "samples", "geometry_mode", "geometry_samples", "steps", "budget_label"
            )
        } for row in selected],
    }}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return stage


def run_alignment_join(
    attack_root: Path,
    composition: Path,
    measurement_root: Path,
    stage: str,
    output: Path,
) -> tuple[subprocess.Popen, Any]:
    python = ALIGNMENT_ROOT / ".venv/bin/python"
    log_path = output.parent / "runner_logs/alignment.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    command = [
        str(python), "scripts/analyze_encoder_alignment.py",
        "--surrogate-results-root", str(attack_root),
        "--composition-config", str(composition),
        "--alignment-metric-map", f"caption_caltech={measurement_root / 'caption_caltech/pairwise_alignment_metrics.csv'}",
        "--alignment-metric-map", f"llava_vqa={measurement_root / 'llava_vqa/pairwise_alignment_metrics.csv'}",
        "--alignment-metric-map", f"receipt_ocr={measurement_root / 'receipt_ocr/pairwise_alignment_metrics.csv'}",
        "--stages", stage, "--output", str(output), "--bootstrap-samples", "2000",
    ]
    process = subprocess.Popen(command, cwd=ALIGNMENT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    return process, log


def primary_only_csv(source: Path, destination: Path, control_target: str) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["target"] != control_target]
        fields = list(rows[0]) if rows else []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _augmentation_name(row: dict[str, Any]) -> str:
    return (
        f"augmentation_{row['mode']}_s{float(row['sigma']):.5f}_n{int(row['samples'])}"
        f"_g{row.get('geometry_mode', 'none')}_gn{int(row.get('geometry_samples', 1))}"
    ).replace(".", "p")


def merge_augmentation_manifest(
    frozen_path: Path,
    dataset: str,
    augmentation_root: Path,
    augmentation_stage: str,
    output: Path,
) -> Path:
    manifest = json.loads(frozen_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    evidence_rows: list[dict[str, Any]] = []
    for path in sorted((augmentation_root / augmentation_stage).glob("*/trial_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "complete" or row.get("dataset") != dataset:
            continue
        name = _augmentation_name(row)
        grouped.setdefault(name, []).append((path, row))
        evidence_rows.append({
            "stage": augmentation_stage, "dataset": dataset,
            "surrogate_set": name, "seed": row["seed"],
            "heldout_macro_asr": row["heldout_macro_asr"],
            "heldout_macro_margin_gain": row.get("heldout_macro_margin_gain"),
            "directory": str(path.parent),
            "selection_source": "heldout_open_source_only",
        })
    if not grouped:
        raise RuntimeError(f"No completed augmentation candidates for {dataset}")
    for rank, (name, values) in enumerate(sorted(grouped.items()), start=len(manifest["candidates"]) + 1):
        baseline = all(value[1]["mode"] == "none" and value[1].get("geometry_mode", "none") == "none" for value in values)
        manifest["candidates"].append({
            "candidate_id": f"{dataset}::{name}",
            "rank_within_dataset": rank,
            "selection_role": "augmentation_baseline" if baseline else "augmentation_candidate",
            "dataset": dataset,
            "surrogate_set": name,
            "seeds": sorted({int(value[1]["seed"]) for value in values}),
            "targets": ["heldout_openclip_macro"],
            "attack_output_directories": [str((value[0].parent / "attack").resolve()) for value in values],
        })

    with Path(manifest["analysis_csv"]).open(newline="", encoding="utf-8") as handle:
        original_rows = list(csv.DictReader(handle))
    combined_rows = [*original_rows, *evidence_rows]
    fields = sorted({key for row in combined_rows for key in row})
    evidence = output.with_suffix(".csv")
    with evidence.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined_rows)
    manifest["analysis_csv"] = str(evidence.resolve())
    manifest["analysis_csv_sha256"] = _sha256(evidence)
    manifest["includes_augmentation_candidates"] = True
    manifest.pop("freeze_hash", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["freeze_hash"] = hashlib.sha256(canonical).hexdigest()
    atomic_write_json(output, manifest)
    return output


def run_api_and_rubric(
    analysis_csv: Path,
    stage: str,
    output: Path,
    run_real: bool,
    limit: int | None,
    augmentation_root: Path | None = None,
    augmentation_stage: str | None = None,
) -> None:
    eval_python = EVAL_ROOT / ".venv/bin/python"
    if not eval_python.is_file():
        eval_python = Path(sys.executable)
    filtered = output / "primary_heldout_results.csv"
    primary_only_csv(analysis_csv, filtered, "dinov2_base_control")
    plan_dir = output / "frozen_api"
    prepare = [
        str(eval_python), str(EVAL_ROOT / "scripts/prepare_task_api_replays.py"),
        "--analysis-csv", str(filtered), "--output", str(plan_dir),
        "--stage", stage, "--budget-mode", "equal_forwards",
        "--baseline", "single_openclip_vit", "--top-k", "2",
        "--minimum-seeds", "2", "--minimum-targets", "5",
    ]
    for dataset, config in API_CONFIGS.items():
        prepare.extend(["--dataset-config", f"{dataset}={config}"])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(EVAL_ROOT / "src"), str(EVAL_ROOT)])
    environment.setdefault("TECHUTOPIA_API_KEY", "research-cycle")
    subprocess.run(prepare, cwd=ROOT, env=environment, check=True)
    replay_manifests = {}
    for dataset in API_CONFIGS:
        frozen = plan_dir / f"frozen_{dataset}.json"
        if augmentation_root is not None and augmentation_stage is not None:
            frozen = merge_augmentation_manifest(
                frozen, dataset, augmentation_root, augmentation_stage,
                plan_dir / f"frozen_combined_{dataset}.json",
            )
        replay_manifests[dataset] = frozen
    if not run_real:
        for dataset, config in API_CONFIGS.items():
            subprocess.run([
                str(eval_python), str(EVAL_ROOT / "scripts/replay_multimodel_eval.py"),
                "--config", str(config), "--frozen-candidates", str(replay_manifests[dataset]),
                "--result-dir", str(output / "api_combined_dry_run" / dataset), "--dry-run",
            ], cwd=ROOT, env=environment, check=True)
        return
    processes = []
    result_dirs = []
    for dataset, config in API_CONFIGS.items():
        result_dir = output / "api_results" / dataset
        result_dirs.append(result_dir)
        command = [
            str(eval_python), str(EVAL_ROOT / "scripts/replay_multimodel_eval.py"),
            "--config", str(config), "--frozen-candidates", str(replay_manifests[dataset]),
            "--result-dir", str(result_dir), "--allow-real-api", "--resume",
        ]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        processes.append((dataset, subprocess.Popen(command, cwd=ROOT, env=environment)))
    failures = [(name, process.wait()) for name, process in processes]
    failures = [value for value in failures if value[1]]
    if failures:
        raise RuntimeError(f"API replay failures: {failures}")
    rubric = [
        str(eval_python), str(EVAL_ROOT / "scripts/build_final_api_rubric.py"),
        "--output", str(output / "api_rubric"),
    ]
    for directory in result_dirs:
        rubric.extend(["--result-dir", str(directory)])
    subprocess.run(rubric, cwd=ROOT, env=environment, check=True)


def api_capability_smoke(output: Path) -> bool:
    eval_python = EVAL_ROOT / ".venv/bin/python"
    if not eval_python.is_file():
        eval_python = Path(sys.executable)
    report = output / "api_capability_smoke.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(EVAL_ROOT / "src"), str(EVAL_ROOT)])
    environment.setdefault("TECHUTOPIA_API_KEY", "research-cycle")
    command = [
        str(eval_python), str(EVAL_ROOT / "scripts/smoke_openai_compatible_vision.py"),
        "--base-url", "https://copilot.techutopia.cn/v1",
        "--model", "gpt-4o-2024-05-13",
        "--api-key-env", "TECHUTOPIA_API_KEY",
        "--image-format", "png", "--image-detail", "low", "--browser-headers",
        "--output", str(report), "--allow-real-api",
    ]
    try:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    except subprocess.CalledProcessError:
        return False
    value = json.loads(report.read_text(encoding="utf-8"))
    return any(
        row.get("model_id") == "gpt-4o-2024-05-13" and row.get("vision_request_ok")
        for row in value.get("results", [])
    )


def plan(spec: dict[str, Any], jobs: int, hours: float, run_api: bool) -> dict[str, Any]:
    return {
        "hours_soft_limit": hours,
        "gpu_jobs": jobs,
        "measurement_models": 11,
        "measurement_datasets": ["caption_caltech", "llava_vqa", "receipt_ocr"],
        "screen_trials": expected_trials(spec, "cross_family_screen"),
        "screen_forward_units_per_item": 1200,
        "augmentation_screen_trials": 27,
        "augmentation_conditions": [
            "none", "gaussian", "uniform", "rademacher", "gaussian_2sample",
            "antithetic_gaussian", "translation", "resize_pad", "gaussian_translation",
        ],
        "alignment_reuses_attack_outputs": True,
        "replication_seeds": spec["stages"]["cross_family_replication"]["seeds"],
        "selection_source": "heldout_open_source_only",
        "dinov2_role": "reported_control_not_primary_score",
        "real_api_after_freeze": run_api,
        "api_models": ["gpt-4o-2024-05-13"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resume-safe cross-family transfer search and frozen API rubric.")
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--hours", type=float, default=9.0)
    parser.add_argument("--gpu-jobs", type=int, default=2)
    parser.add_argument("--augmentation-workers", type=int, default=1)
    parser.add_argument("--promote", type=int, default=3)
    parser.add_argument("--skip-measurement", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--api-limit", type=int)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    spec_path = Path(args.spec).resolve()
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if args.plan:
        print(json.dumps(plan(spec, args.gpu_jobs, args.hours, args.run_api), indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible in this process. Run this script from the host GPU shell.")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = output / "cycle_state.json"
    cache_dir = ROOT.parent / "attack-vllm-noise/models/open_clip"
    deadline = time.monotonic() + args.hours * 3600
    write_state(state, status="starting", plan=plan(spec, args.gpu_jobs, args.hours, args.run_api))

    api_available = False
    if args.run_api:
        write_state(state, status="checking_api_capability")
        api_available = api_capability_smoke(output)
        write_state(state, status="api_capability_checked", api_available=api_available)

    if not args.skip_measurement:
        write_state(state, status="measuring_cross_family_geometry")
        run_measurements(spec, output, cache_dir)

    augmentation_spec = GEOMETRIC_ROOT / "configs/research_cycle_augmentation.yaml"
    augmentation_process, augmentation_log = start_augmentation_stage(
        augmentation_spec, "strict_equal_forward_screen", output,
        args.augmentation_workers,
    )
    screen_root = output / "attacks"
    run_bounded_stage(spec, spec_path, "cross_family_screen", screen_root, args.gpu_jobs, cache_dir, state, deadline)
    analysis_csv = run_analysis(screen_root, output / "analysis_screen")
    screen_selection = select_and_write(
        analysis_csv, ROOT / spec["composition_config"], "cross_family_screen",
        output / "screen_selection.json", args.promote,
    )
    frozen_path = output / "frozen_replication_spec.yaml"
    frozen = write_replication_spec(spec, screen_selection["promoted"], frozen_path)

    wait_process(augmentation_process, augmentation_log, "augmentation screening")
    augmentation_root = output / "augmentation"
    augmentation_selected = select_augmentations(
        augmentation_root, "strict_equal_forward_screen",
        output / "augmentation_screen_selection.json",
    )
    augmentation_replication_spec = output / "frozen_augmentation_replication.yaml"
    augmentation_replication_stage = write_augmentation_replication_spec(
        augmentation_spec, augmentation_selected, augmentation_replication_spec,
    )

    if time.monotonic() >= deadline:
        write_state(
            state, status="screen_complete_deadline_reached",
            promoted=screen_selection["promoted"],
            augmentation_promoted=augmentation_selected,
        )
        return

    alignment_process, alignment_log = run_alignment_join(
        screen_root, ROOT / spec["composition_config"], output / "alignment",
        "cross_family_screen", output / "alignment_analysis",
    )
    augmentation_replication_process, augmentation_replication_log = start_augmentation_stage(
        augmentation_replication_spec, augmentation_replication_stage, output,
        args.augmentation_workers,
    )
    run_bounded_stage(frozen, frozen_path, "cross_family_replication", screen_root, args.gpu_jobs, cache_dir, state, deadline)
    wait_process(alignment_process, alignment_log, "alignment analysis")
    wait_process(augmentation_replication_process, augmentation_replication_log, "augmentation replication")
    augmentation_final = select_augmentations(
        augmentation_root, augmentation_replication_stage,
        output / "augmentation_final_selection.json",
    )
    final_analysis = run_analysis(screen_root, output / "analysis_final")
    final_selection = select_and_write(
        final_analysis, ROOT / spec["composition_config"], "cross_family_replication",
        output / "final_selection.json", args.promote,
    )
    write_state(
        state, status="candidates_frozen", promoted=final_selection["promoted"],
        augmentation_promoted=augmentation_final,
    )
    run_api_and_rubric(
        final_analysis, "cross_family_replication", output, args.run_api and api_available, args.api_limit,
        augmentation_root=augmentation_root,
        augmentation_stage=augmentation_replication_stage,
    )
    write_state(
        state, status="complete", promoted=final_selection["promoted"],
        augmentation_promoted=augmentation_final,
        api_requested=args.run_api, api_available=api_available,
        api_executed=args.run_api and api_available, api_used_for_selection=False,
    )


if __name__ == "__main__":
    main()
