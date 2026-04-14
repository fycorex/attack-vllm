#!/usr/bin/env python3
"""
Optimization loop for transfer attack success rate.

Iteratively adjusts attack hyperparameters to maximize ASR against
target victim model (e.g., local ollama QwenVL).

Usage:
    PYTHONPATH=src python scripts/run_optimization_loop.py \
        --config configs/ollama_test.yaml \
        --target_asr 0.5 \
        --max_iterations 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attack_vlm_repro.config import load_config, AttackConfig
from attack_vlm_repro.attack import run_attack


# Focused hyperparameter search space (per paper insights)
SEARCH_SPACE = {
    "surrogate_count": [2, 3, 5],
    "augmentation_batches": [1, 2, 4],
}

# Fixed parameters (paper defaults)
FIXED_PARAMS = {
    "epsilon": 16.0 / 255.0,
    "temperature": 0.1,
    "top_k": 4,
}

# Diverse surrogate pools
SURROGATE_POOLS = {
    2: ["ViT-B-32:laion2b_s34b_b79k", "RN50:openai"],  # ViT + ResNet
    3: ["ViT-B-32:laion2b_s34b_b79k", "RN50:openai", "ViT-L-14:openai"],  # 2 ViT + ResNet
    5: ["ViT-B-32:laion2b_s34b_b79k", "ViT-B-16:laion2b_s34b_b88k", "RN50:openai", "RN101:openai", "ViT-L-14:openai"],
}


def calculate_ensemble_diversity(surrogates: list) -> float:
    """Compute architecture diversity score."""
    architectures = set()
    for s in surrogates:
        name = s if isinstance(s, str) else s.get("model_name", "")
        if "ViT" in name:
            architectures.add("ViT")
        elif "RN" in name:
            architectures.add("ResNet")
    return len(architectures)


def apply_iteration_params(config: AttackConfig, iteration: int) -> None:
    """Apply hyperparameters for current iteration."""
    # Cycle through surrogate counts
    surrogate_idx = iteration % len(SEARCH_SPACE["surrogate_count"])
    surrogate_count = SEARCH_SPACE["surrogate_count"][surrogate_idx]

    # Cycle through augmentation batches
    aug_idx = (iteration // len(SEARCH_SPACE["surrogate_count"])) % len(SEARCH_SPACE["augmentation_batches"])
    aug_batches = SEARCH_SPACE["augmentation_batches"][aug_idx]

    # Apply surrogate pool
    pool = SURROGATE_POOLS.get(surrogate_count, SURROGATE_POOLS[2])
    for i, surrogate in enumerate(config.surrogates):
        if i < len(pool):
            parts = pool[i].split(":")
            surrogate.model_name = parts[0]
            surrogate.pretrained = parts[1] if len(parts) > 1 else "openai"
            surrogate.enabled = True
        else:
            surrogate.enabled = False

    # Apply augmentation batches
    config.attack.augmentation_batches = aug_batches

    print(f"  Surrogate count: {surrogate_count}, Aug batches: {aug_batches}")


def calculate_asr(results: list[dict], victim_type: str = "ollama") -> dict:
    """Calculate Attack Success Rate for a specific victim type."""
    eval_key = f"{victim_type}_eval"
    success_key = f"{victim_type}_success"

    completed = [
        r for r in results
        if r.get(eval_key) is not None and not r.get(eval_key, {}).get("evaluation_failed")
    ]
    failed = [
        r for r in results
        if r.get(eval_key, {}).get("evaluation_failed")
    ]
    successes = sum(1 for r in completed if r.get(eval_key, {}).get(success_key, False))

    return {
        "total_items": len(results),
        "completed_evaluations": len(completed),
        "failed_evaluations": len(failed),
        "successful_attacks": successes,
        "asr": successes / len(completed) if completed else 0.0,
    }


def run_optimization_loop(
    config_path: str,
    target_asr: float = 0.5,
    max_iterations: int = 10,
    output_root: str = "outputs/optimization_loop",
) -> list[dict]:
    """Iteratively optimize attack for target ASR."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_log = []

    for iteration in range(max_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{max_iterations}")
        print(f"{'='*50}")

        # Load fresh config
        config = load_config(config_path)
        config.paths.output_dir = str(output_dir / f"iteration_{iteration:02d}")

        # Apply iteration hyperparameters
        apply_iteration_params(config, iteration)

        # Calculate diversity
        enabled_surrogates = [s for s in config.surrogates if s.enabled]
        diversity = calculate_ensemble_diversity(enabled_surrogates)

        # Run attack
        print(f"  Running attack with {len(enabled_surrogates)} surrogates (diversity: {diversity})...")
        start_time = time.time()
        attack_results = run_attack(config)
        elapsed = time.time() - start_time

        # Calculate ASR
        asr_info = calculate_asr(attack_results.get("items", []), victim_type="ollama")
        proxy_asr = attack_results.get("proxy_success_rate", 0.0)

        # Log iteration
        iteration_log = {
            "iteration": iteration,
            "surrogate_count": len(enabled_surrogates),
            "ensemble_diversity": diversity,
            "augmentation_batches": config.attack.augmentation_batches,
            "proxy_asr": proxy_asr,
            "ollama_asr": asr_info["asr"],
            "completed_evaluations": asr_info["completed_evaluations"],
            "elapsed_seconds": elapsed,
        }
        results_log.append(iteration_log)

        print(f"  Proxy ASR: {proxy_asr:.1%}")
        print(f"  Ollama ASR: {asr_info['asr']:.1%} ({asr_info['successful_attacks']}/{asr_info['completed_evaluations']})")
        print(f"  Elapsed: {elapsed:.1f}s")

        # Check termination
        if asr_info["asr"] >= target_asr:
            print(f"\n*** Target ASR {target_asr:.0%} achieved! ***")
            break

    # Save optimization log
    log_path = output_dir / "optimization_log.json"
    log_path.write_text(json.dumps(results_log, indent=2), encoding="utf-8")
    print(f"\nOptimization log saved to {log_path}")

    # Find best iteration
    best = max(results_log, key=lambda x: x["ollama_asr"])
    print(f"\nBest iteration: {best['iteration']} with ASR {best['ollama_asr']:.1%}")

    return results_log


def main():
    parser = argparse.ArgumentParser(description="Optimization loop for transfer attack ASR")
    parser.add_argument("--config", required=True, help="Path to base config file")
    parser.add_argument("--target_asr", type=float, default=0.5, help="Target ASR to achieve")
    parser.add_argument("--max_iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--output_root", default="outputs/optimization_loop", help="Output directory")
    args = parser.parse_args()

    run_optimization_loop(
        config_path=args.config,
        target_asr=args.target_asr,
        max_iterations=args.max_iterations,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
