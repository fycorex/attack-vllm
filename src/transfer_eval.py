from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
import time
from typing import Any

import yaml
from PIL import Image

from api_victims import create_api_victim


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    enabled: bool = True
    api_key_env: str | None = None
    base_url: str | None = None
    max_output_tokens: int | None = 64
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    reasoning_effort: str | None = None
    timeout_seconds: float = 90
    max_retries: int = 4
    retry_backoff_seconds: float = 2
    requests_per_minute: float | None = None
    estimated_cost_per_request: float | None = None
    image_format: str = "png"
    jpeg_quality: int = 95
    image_detail: str | None = None
    browser_headers: bool = False

    def effective(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and k != "api_key_env"}


@dataclass
class ReplayConfig:
    experiment_name: str
    task_type: str = "caption"
    caption_prompt: str = "Provide a concise description of the image using no more than three sentences."
    question_fallback: str = "What is the main object in the image?"
    require_source_absent: bool = True
    bootstrap_samples: int = 2000
    models: list[ModelConfig] = field(default_factory=list)


def load_replay_config(path: str | Path) -> ReplayConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = set(ModelConfig.__dataclass_fields__)
    models = []
    for entry in raw.get("models", []):
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
        models.append(ModelConfig(**entry))
    if not models:
        raise ValueError("Replay config must define at least one model")
    return ReplayConfig(
        experiment_name=raw.get("experiment_name", Path(path).stem),
        task_type=raw.get("task_type", "caption"),
        caption_prompt=raw.get("caption_prompt", ReplayConfig.caption_prompt),
        question_fallback=raw.get("question_fallback", ReplayConfig.question_fallback),
        require_source_absent=bool(raw.get("require_source_absent", True)),
        bootstrap_samples=int(raw.get("bootstrap_samples", 2000)),
        models=models,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_candidate_manifest(path: str | Path) -> tuple[list[Path], dict[str, Any]]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = value.get("freeze_hash")
    canonical = {key: item for key, item in value.items() if key != "freeze_hash"}
    actual_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if claimed_hash != actual_hash:
        raise ValueError("Frozen candidate manifest hash mismatch")
    if value.get("selection_source") != "heldout_open_source_only" or value.get("api_results_used_for_selection") is not False:
        raise ValueError("Frozen candidates must be selected from held-out open-source results only")
    analysis = Path(value["analysis_csv"])
    if not analysis.is_file() or sha256_file(analysis) != value.get("analysis_csv_sha256"):
        raise ValueError("Frozen candidate analysis CSV is missing or changed")
    directories = []
    directory_metadata: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(value.get("candidates") or [], 1):
        candidate_id = str(candidate.get("candidate_id") or
                           f"{candidate.get('dataset', 'dataset')}::{candidate.get('surrogate_set', 'candidate')}::{candidate.get('rank_within_dataset', index)}")
        metadata = {"candidate_id": candidate_id, "candidate_dataset": candidate.get("dataset"),
                    "candidate_method": candidate.get("surrogate_set"),
                    "candidate_rank": candidate.get("rank_within_dataset"),
                    "candidate_selection_role": candidate.get("selection_role")}
        for item in candidate.get("attack_output_directories") or []:
            directory = Path(item).resolve(); directories.append(directory)
            prior = directory_metadata.get(str(directory))
            if prior is not None and prior != metadata:
                raise ValueError(f"Frozen attack directory belongs to multiple candidates: {directory}")
            directory_metadata[str(directory)] = metadata
    directories = list(dict.fromkeys(directory.resolve() for directory in directories))
    if not directories or any(not directory.is_dir() for directory in directories):
        raise ValueError("Frozen candidate attack output directories are missing")
    return directories, {"path": str(path.resolve()), "freeze_hash": claimed_hash,
                         "analysis_csv_sha256": value["analysis_csv_sha256"], "candidate_count": len(value["candidates"]),
                         "directory_metadata": directory_metadata}


def stable_cache_key(provider: str, model_id: str, prompt: str, image_sha256: str, effective_config: dict) -> str:
    value = {"schema_version": 1, "provider": provider, "model_id": model_id, "prompt": prompt,
             "image_sha256": image_sha256, "effective_config": effective_config}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_record_key(request_cache_key: str, evaluation_sample_id: str, condition: str) -> str:
    value = {"schema_version": 1, "request_cache_key": request_cache_key,
             "evaluation_sample_id": evaluation_sample_id, "condition": condition}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evaluation_sample_id(metrics_path: Path, item_id: str) -> str:
    source = str(metrics_path.parent.parent.resolve())
    digest = hashlib.sha256(f"{source}\0{item_id}".encode()).hexdigest()[:16]
    return f"{item_id}::{digest}"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def discover_items(output_dirs: list[Path], limit: int | None = None) -> list[Path]:
    found: list[Path] = []
    for root in output_dirs:
        for metrics in sorted(root.rglob("metrics.json")):
            if metrics.with_name("clean.png").is_file() and metrics.with_name("adversarial.png").is_file():
                found.append(metrics)
    unique = list(dict.fromkeys(p.resolve() for p in found))
    return unique[:limit] if limit is not None else unique


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def _contains(text: str, values: list[str]) -> bool:
    normalized = _norm(text)
    return any(re.search(rf"(?<!\w){re.escape(_norm(v))}(?!\w)", normalized) for v in values if _norm(v))


def score_output(text: str, metrics: dict, target: bool, require_source_absent: bool = True) -> dict[str, bool]:
    prefix = "target" if target else "source"
    exact = metrics.get(f"{prefix}_answer_text")
    keys = metrics.get(f"{prefix}_answer_keywords") or metrics.get(f"{prefix}_text_keywords") or metrics.get(f"{prefix}_keywords") or [metrics.get(f"{prefix}_label", "")]
    opposite = "source" if target else "target"
    opposite_keys = metrics.get(f"{opposite}_answer_keywords") or metrics.get(f"{opposite}_keywords") or [metrics.get(f"{opposite}_label", "")]
    normalized = _norm(text)
    success = bool(exact and normalized == _norm(exact)) or _contains(text, keys)
    source_present = _contains(text, opposite_keys if target else keys)
    if target and require_source_absent and source_present:
        success = False
    return {"success": success, "source_present": source_present}


def conditioned_success(clean_target_success: bool, adversarial_target_success: bool) -> bool:
    return bool(adversarial_target_success and not clean_target_success)


def git_state(cwd: Path) -> dict[str, Any]:
    def run(*args): return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False).stdout.strip()
    return {"commit_sha": run("git", "rev-parse", "HEAD") or None, "dirty": bool(run("git", "status", "--porcelain"))}


def _prompt(cfg: ReplayConfig, metrics: dict) -> str:
    if cfg.task_type.lower() == "caption": return cfg.caption_prompt
    return metrics.get("question") or cfg.question_fallback


def _completed(path: Path) -> set[str]:
    if not path.exists(): return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try: result.add(json.loads(line)["cache_key"])
        except (ValueError, KeyError): continue
    return result


def summarize(records: list[dict], bootstrap_samples: int = 2000) -> dict:
    attempted = len(records); failures = sum(bool(r.get("error")) for r in records)
    refusals = sum(bool(r.get("refusal")) for r in records)
    pairs: dict[tuple, dict] = {}
    for row in records:
        sample_id = row.get("evaluation_sample_id", row["item_id"])
        pairs.setdefault((row["provider"], row["model_id"], sample_id), {})[row["condition"]] = row
    paired = [p for p in pairs.values() if "clean" in p and "adversarial" in p]
    valid = [p for p in paired if not p["clean"].get("error") and not p["adversarial"].get("error")]
    cond = [conditioned_success(p["clean"].get("target_success", False), p["adversarial"].get("target_success", False)) for p in paired]
    rng = random.Random(0); estimates = []
    if cond:
        for _ in range(bootstrap_samples): estimates.append(sum(rng.choice(cond) for _ in cond) / len(cond))
        estimates.sort(); ci = [estimates[int(.025 * (len(estimates)-1))], estimates[int(.975 * (len(estimates)-1))]]
    else: ci = [None, None]
    denominator = len(paired)
    actual_requests = sum(not row.get("response_reused", False) for row in records)
    return {"attempted_requests": attempted, "actual_api_requests": actual_requests,
            "reused_response_records": attempted-actual_requests,
            "valid_requests": attempted-failures, "attempted_pairs": len(pairs),
            "paired_item_count": denominator, "valid_pair_count": len(valid), "api_failure_rate": failures/attempted if attempted else 0,
            "refusal_rate": refusals/attempted if attempted else 0, "clean_target_success_rate": sum(p["clean"].get("target_success", False) for p in paired)/denominator if denominator else 0,
            "adversarial_target_success_rate": sum(p["adversarial"].get("target_success", False) for p in paired)/denominator if denominator else 0,
            "conditioned_asr": sum(cond)/denominator if denominator else 0, "conditioned_asr_ci95": ci,
            "source_suppression_rate": sum(not p["adversarial"].get("source_present", False) for p in paired)/denominator if denominator else 0}


def run_replay(config_path: Path, output_dirs: list[Path], result_dir: Path, *, allow_real_api: bool = False,
               dry_run: bool = False, limit: int | None = None, resume: bool = False,
               max_requests: int | None = None, max_estimated_cost: float | None = None, transports: dict | None = None,
               frozen_candidate_manifest: Path | None = None) -> dict:
    frozen_info = None
    if frozen_candidate_manifest is not None:
        if output_dirs:
            raise ValueError("Use either output directories or a frozen candidate manifest, not both")
        output_dirs, frozen_info = load_frozen_candidate_manifest(frozen_candidate_manifest)
    if not output_dirs:
        raise ValueError("At least one output directory or frozen candidate manifest is required")
    cfg = load_replay_config(config_path); items = discover_items(output_dirs, limit)
    models = [m for m in cfg.models if m.enabled]
    unique_requests: set[str] = set()
    for model in models:
        for metrics_path in items:
            metrics = json.loads(metrics_path.read_text()); prompt = _prompt(cfg, metrics)
            for condition in ("clean", "adversarial"):
                image_hash = sha256_file(metrics_path.with_name(f"{condition}.png"))
                unique_requests.add(stable_cache_key(model.provider, model.model_id, prompt, image_hash, model.effective()))
    count = len(unique_requests); record_count = len(items) * len(models) * 2
    average_cost = sum((model.estimated_cost_per_request or 0) for model in models) / len(models) if models else 0
    cost = count * average_cost
    estimate = {"items": len(items), "models": len(models), "evaluation_records": record_count,
                "estimated_requests": count, "deduplicated_requests": record_count-count, "estimated_cost": cost}
    if max_requests is not None and count > max_requests: raise RuntimeError(f"Estimated {count} requests exceeds --max-requests {max_requests}")
    if max_estimated_cost is not None and cost > max_estimated_cost: raise RuntimeError(f"Estimated cost {cost:.4f} exceeds limit {max_estimated_cost:.4f}")
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = {**estimate, "experiment_name": cfg.experiment_name, "source_output_directories": [str(p.resolve()) for p in output_dirs],
                "config": str(config_path.resolve()), "created_at": datetime.now(timezone.utc).isoformat(), **git_state(Path.cwd()), "real_api": bool(allow_real_api and not dry_run)}
    if frozen_info is not None:
        manifest["frozen_candidates"] = frozen_info
    atomic_write_json(result_dir / "run_manifest.json", manifest)
    if dry_run or not allow_real_api: return {**estimate, "dry_run": True}
    records_path = result_dir / "requests.jsonl"; done = _completed(records_path) if resume else set(); records = []
    if resume and records_path.exists(): records = [json.loads(x) for x in records_path.read_text().splitlines() if x.strip()]
    request_cache = {row["request_cache_key"]: row for row in records
                     if row.get("request_cache_key") and not row.get("error")}
    for model in models:
        victim = create_api_victim(asdict(model), transport=(transports or {}).get(model.provider))
        last = 0.0
        try:
            for metrics_path in items:
                metrics = json.loads(metrics_path.read_text()); prompt = _prompt(cfg, metrics)
                item_id = str(metrics.get("item_id", metrics_path.parent.name))
                sample_id = evaluation_sample_id(metrics_path, item_id)
                for condition in ("clean", "adversarial"):
                    image_path = metrics_path.with_name(f"{condition}.png"); image_hash = sha256_file(image_path)
                    effective = model.effective()
                    request_key = stable_cache_key(model.provider, model.model_id, prompt, image_hash, effective)
                    key = stable_record_key(request_key, sample_id, condition)
                    if key in done: continue
                    retry = 0; response = None; err = None; reused = request_key in request_cache
                    if reused:
                        cached = request_cache[request_key]
                        response_data = {"text": cached.get("response", ""),
                                         "resolved_model_id": cached.get("resolved_model_id"),
                                         "refusal": cached.get("refusal", False),
                                         "finish_reason": cached.get("finish_reason"),
                                         "error": cached.get("error")}
                    else:
                        while retry <= model.max_retries:
                            try:
                                if model.requests_per_minute:
                                    wait = 60/model.requests_per_minute - (time.monotonic()-last)
                                    if wait > 0: time.sleep(wait)
                                response = victim.generate(Image.open(image_path).convert("RGB"), prompt); last = time.monotonic(); break
                            except Exception as exc:
                                err = {"type": type(exc).__name__, "message": str(exc)}
                                if retry == model.max_retries: break
                                time.sleep(min(model.retry_backoff_seconds * (2 ** retry), 60)); retry += 1
                        response_data = response.to_dict() if response else {}
                    scored = score_output(response_data.get("text", ""), metrics, True, cfg.require_source_absent)
                    source_directory = str(metrics_path.parent.parent.resolve())
                    candidate = (frozen_info or {}).get("directory_metadata", {}).get(source_directory, {})
                    record = {"cache_key": key, "request_cache_key": request_key,
                              "evaluation_sample_id": sample_id, "response_reused": reused,
                              "experiment_name": cfg.experiment_name, "source_output_directory": source_directory,
                              "candidate_id": candidate.get("candidate_id", source_directory),
                              "candidate_dataset": candidate.get("candidate_dataset"),
                              "candidate_method": candidate.get("candidate_method"),
                              "candidate_rank": candidate.get("candidate_rank"),
                              "candidate_selection_role": candidate.get("candidate_selection_role"),
                              "item_id": item_id, "task_type": cfg.task_type, "provider": model.provider,
                              "model_id": model.model_id, "resolved_model_id": response_data.get("resolved_model_id"), "request_timestamp": datetime.now(timezone.utc).isoformat(),
                              "prompt": prompt, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(), "image_sha256": image_hash, "condition": condition,
                              "effective_evaluation_config": effective, "response": response_data.get("text", ""), "latency_seconds": 0 if reused else response_data.get("latency_seconds", 0),
                              "usage": {} if reused else response_data.get("usage", {}), "retry_count": retry, "error": err or response_data.get("error"),
                              "refusal": bool(response_data.get("refusal")), "finish_reason": response_data.get("finish_reason"),
                              "target_success": scored["success"], "source_present": scored["source_present"], "git_commit_sha": manifest["commit_sha"], "repository_dirty": manifest["dirty"]}
                    append_jsonl(records_path, record); records.append(record); done.add(key)
                    if not record.get("error"): request_cache.setdefault(request_key, record)
        finally: victim.close()
    by_model = {}
    for model in models:
        subset = [r for r in records if r["provider"] == model.provider and r["model_id"] == model.model_id]
        value = summarize(subset, cfg.bootstrap_samples); by_model[f"{model.provider}__{model.model_id}"] = value
        atomic_write_json(result_dir / f"summary_{model.provider}_{model.model_id.replace('/', '_')}.json", value)
    by_candidate_model = {}
    for candidate_id in sorted({str(row["candidate_id"]) for row in records}):
        for model in models:
            subset = [row for row in records if str(row["candidate_id"]) == candidate_id and
                      row["provider"] == model.provider and row["model_id"] == model.model_id]
            if subset:
                key = f"{candidate_id}::{model.provider}::{model.model_id}"
                by_candidate_model[key] = {"candidate_id": candidate_id,
                    "candidate_dataset": subset[0].get("candidate_dataset"),
                    "candidate_method": subset[0].get("candidate_method"),
                    "candidate_selection_role": subset[0].get("candidate_selection_role"),
                    "provider": model.provider, "model_id": model.model_id, **summarize(subset, cfg.bootstrap_samples)}
    combined = {"models": by_model, "candidate_models": by_candidate_model,
                "macro_conditioned_asr": sum(v["conditioned_asr"] for v in by_model.values())/len(by_model) if by_model else 0,
                **summarize(records, cfg.bootstrap_samples)}
    atomic_write_json(result_dir / "summary_combined.json", combined)
    with (result_dir / "summary_combined.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *next(iter(by_model.values())).keys()] if by_model else ["model"]); writer.writeheader()
        for name, value in by_model.items(): writer.writerow({"model": name, **value})
    if by_candidate_model:
        with (result_dir / "summary_by_candidate_model.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(next(iter(by_candidate_model.values())).keys())
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(by_candidate_model.values())
    return {**estimate, "dry_run": False, "output": str(result_dir)}
