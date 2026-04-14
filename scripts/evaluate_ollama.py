#!/usr/bin/env python3
"""Evaluate generated adversarial images against ollama victim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image
from attack_vlm_repro.config import OllamaVictimConfig
from attack_vlm_repro.ollama_victim import OllamaVictim
from attack_vlm_repro.data import load_manifest


def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial images against ollama")
    parser.add_argument("--output_dir", required=True, help="Attack output directory")
    parser.add_argument("--manifest", default="data/cifar10_caption_attack_demo/manifest.json")
    parser.add_argument("--model", default="qwen3-vl:4b")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest = load_manifest(args.manifest)

    config = OllamaVictimConfig(enabled=True, model_name=args.model)
    victim = OllamaVictim(config)

    # Health check
    health = victim.health_check()
    print(f"Ollama health: {health}")
    if not health["healthy"]:
        print("ERROR: Ollama not healthy!")
        return

    results = []
    for item in manifest.items:
        item_dir = output_dir / item.item_id
        clean_path = item_dir / "clean.png"
        adv_path = item_dir / "adversarial.png"

        if not clean_path.exists() or not adv_path.exists():
            print(f"Skipping {item.item_id}: images not found")
            continue

        clean_img = Image.open(clean_path)
        adv_img = Image.open(adv_path)

        result = victim.evaluate(clean_img, adv_img, item)
        results.append({
            "item_id": item.item_id,
            "source_label": item.source_label,
            "target_label": item.target_label,
            **result
        })

        status = "SUCCESS" if result.get("ollama_success") else "FAIL"
        print(f"{item.item_id}: {status}")
        print(f"  Clean: {result.get('clean_output', '')[:60]}...")
        print(f"  Adv:   {result.get('adversarial_output', '')[:60]}...")

    # Summary
    successes = sum(1 for r in results if r.get("ollama_success"))
    asr = successes / len(results) if results else 0.0
    print(f"\n=== Summary ===")
    print(f"ASR: {asr:.1%} ({successes}/{len(results)})")

    # Save results
    (output_dir / "ollama_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {output_dir / 'ollama_eval.json'}")


if __name__ == "__main__":
    main()
