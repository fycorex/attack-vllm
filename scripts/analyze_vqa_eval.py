#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CATEGORY_LABELS = {
    "conv": "Conversation",
    "detail": "Detail",
    "complex": "Reasoning",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    total = len(rows)
    success = sum(1 for row in rows if row.get("gpt_success") is True)
    summary["overall"] = {
        "success_count": success,
        "total": total,
        "attack_success_rate": float(success / total) if total else None,
    }
    for raw_category, label in CATEGORY_LABELS.items():
        category_rows = [row for row in rows if row.get("question_category") == raw_category]
        category_total = len(category_rows)
        category_success = sum(1 for row in category_rows if row.get("gpt_success") is True)
        summary[label] = {
            "success_count": category_success,
            "total": category_total,
            "attack_success_rate": float(category_success / category_total) if category_total else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LLaVA-Bench VQA replay results by question category.")
    parser.add_argument("jsonl", nargs="+", help="Evaluation JSONL files to summarize.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    payload: dict[str, Any] = {}
    for raw_path in args.jsonl:
        path = Path(raw_path)
        payload[path.stem] = summarize_rows(load_jsonl(path))

    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
