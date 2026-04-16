#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    total = len(rows)
    success = sum(1 for row in rows if row.get("gpt_success") is True)
    output["overall"] = {
        "success_count": success,
        "total": total,
        "attack_success_rate": float(success / total) if total else None,
    }
    question_types = sorted({row.get("question_type") or "unknown" for row in rows})
    for question_type in question_types:
        subset = [row for row in rows if (row.get("question_type") or "unknown") == question_type]
        subset_total = len(subset)
        subset_success = sum(1 for row in subset if row.get("gpt_success") is True)
        output[question_type] = {
            "success_count": subset_success,
            "total": subset_total,
            "attack_success_rate": float(subset_success / subset_total) if subset_total else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize receipt text replay results by question type.")
    parser.add_argument("jsonl", nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    payload = {Path(path).stem: summarize(load_jsonl(Path(path))) for path in args.jsonl}
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
