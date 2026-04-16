#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from config import load_config
from data import AttackItem
from gpt_victim import GPTVictim


def build_item(metrics: dict) -> AttackItem:
    return AttackItem(
        item_id=metrics["item_id"],
        image_path=Path(metrics["image_path"]),
        source_label=metrics["source_label"],
        target_label=metrics["target_label"],
        positive_image_paths=[],
        negative_image_paths=[],
        source_keywords=metrics["source_keywords"],
        target_keywords=metrics["target_keywords"],
        question=metrics.get("question"),
        source_answer_text=metrics.get("source_answer_text"),
        target_answer_text=metrics.get("target_answer_text"),
        source_answer_keywords=metrics.get("source_answer_keywords") or metrics["source_keywords"],
        target_answer_keywords=metrics.get("target_answer_keywords") or metrics["target_keywords"],
        source_text_keywords=metrics.get("source_text_keywords") or metrics["source_keywords"],
        target_text_keywords=metrics.get("target_text_keywords") or metrics["target_keywords"],
        metadata=metrics.get("metadata", {}),
    )


def load_metrics_paths(explicit_paths: list[str], glob_pattern: str | None, limit: int | None) -> list[Path]:
    paths: list[Path] = [Path(path) for path in explicit_paths]
    if glob_pattern:
        paths.extend(sorted(Path().glob(glob_pattern)))
    deduped = []
    seen = set()
    for path in paths:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    if limit is not None:
        deduped = deduped[:limit]
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay GPT evaluation on existing clean/adversarial image pairs.")
    parser.add_argument("--config", required=True, help="GPT YAML config to use for evaluation.")
    parser.add_argument(
        "--metrics",
        action="append",
        default=[],
        help="Path to an existing item metrics.json. Can be passed multiple times.",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Optional glob for metrics paths, for example 'outputs/paper_caltech/item_*/metrics.json'.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit after expanding inputs.")
    args = parser.parse_args()

    metrics_paths = load_metrics_paths(args.metrics, args.glob, args.limit)
    if not metrics_paths:
        raise SystemExit("No metrics paths selected.")

    cfg = load_config(args.config)
    victim = GPTVictim(cfg.evaluation.gpt_victim)

    try:
        for metrics_path in metrics_paths:
            metrics = json.loads(metrics_path.read_text())
            item = build_item(metrics)
            result = victim.evaluate(
                Image.open(metrics_path.with_name("clean.png")).convert("RGB"),
                Image.open(metrics_path.with_name("adversarial.png")).convert("RGB"),
                item,
            )
            print(
                json.dumps(
                    {
                        "metrics_path": str(metrics_path),
                        "source_label": item.source_label,
                        "target_label": item.target_label,
                        "question_category": item.metadata.get("question_category"),
                        "question_category_name": item.metadata.get("question_category_name"),
                        "question_type": item.metadata.get("question_type"),
                        "bbox_label": item.metadata.get("bbox_label"),
                        "receipt_image_name": item.metadata.get("image_name"),
                        "question": item.question,
                        "source_answer_text": item.source_answer_text,
                        "target_answer_text": item.target_answer_text,
                        "gpt_success": result.get("gpt_success"),
                        "clean_output": result.get("clean_output"),
                        "adversarial_output": result.get("adversarial_output"),
                    },
                    ensure_ascii=True,
                )
            )
    finally:
        victim.unload()


if __name__ == "__main__":
    main()
