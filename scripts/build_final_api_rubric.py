#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from transfer_eval import conditioned_success, summarize


RUBRIC_FIELDS = [
    "task_type", "candidate_method", "candidate_selection_role", "provider", "model_id",
    "attempted_requests", "valid_requests", "paired_item_count", "valid_pair_count",
    "clean_target_success_rate", "adversarial_target_success_rate", "conditioned_asr",
    "conditioned_asr_ci95_low", "conditioned_asr_ci95_high", "source_suppression_rate",
    "refusal_rate", "api_failure_rate",
]


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_records(result_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for directory in result_dirs:
        path = directory / "requests.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing replay records: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rubric_rows(records: list[dict[str, Any]], bootstrap_samples: int = 2000) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (
            str(row.get("task_type", "unknown")),
            str(row.get("candidate_method") or row.get("candidate_id") or "unknown"),
            str(row.get("candidate_selection_role") or "frozen"),
            str(row["provider"]),
            str(row["model_id"]),
        )
        grouped[key].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        summary = summarize(values, bootstrap_samples)
        ci = summary.pop("conditioned_asr_ci95")
        output.append({
            "task_type": key[0], "candidate_method": key[1],
            "candidate_selection_role": key[2], "provider": key[3], "model_id": key[4],
            **summary,
            "conditioned_asr_ci95_low": ci[0],
            "conditioned_asr_ci95_high": ci[1],
        })
    return output


def item_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        sample = str(row.get("evaluation_sample_id") or row.get("item_id"))
        key = (
            str(row.get("task_type", "unknown")),
            str(row.get("candidate_method") or row.get("candidate_id") or "unknown"),
            str(row["provider"]), str(row["model_id"]), sample,
        )
        pairs[key][str(row["condition"])] = row
    output = []
    for key, pair in sorted(pairs.items()):
        clean, adversarial = pair.get("clean"), pair.get("adversarial")
        complete = clean is not None and adversarial is not None
        valid = complete and not clean.get("error") and not adversarial.get("error")
        output.append({
            "task_type": key[0], "candidate_method": key[1], "provider": key[2],
            "model_id": key[3], "evaluation_sample_id": key[4],
            "pair_complete": complete, "pair_valid": bool(valid),
            "clean_target_success": bool(clean and clean.get("target_success")),
            "adversarial_target_success": bool(adversarial and adversarial.get("target_success")),
            "conditioned_success": bool(complete and conditioned_success(
                bool(clean.get("target_success")), bool(adversarial.get("target_success")))),
            "clean_refusal": bool(clean and clean.get("refusal")),
            "adversarial_refusal": bool(adversarial and adversarial.get("refusal")),
            "clean_error": bool(clean and clean.get("error")),
            "adversarial_error": bool(adversarial and adversarial.get("error")),
        })
    return output


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{100 * value:.1f}%"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Frozen API Evaluation Rubric", "",
        "This report is descriptive external validation. API outcomes were not used for candidate selection or tuning.", "",
        "Rubric dimensions: clean target false-positive rate, adversarial target rate, conditional ASR, source suppression, refusal rate, failure rate, and paired sample coverage.", "",
    ]
    for task in sorted({row["task_type"] for row in rows}):
        lines.extend([f"## {task}", "", "| Candidate | Model | Paired | Clean target | Adv target | Conditional ASR | Source suppression | Refusal | Failure |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
        for row in (value for value in rows if value["task_type"] == task):
            lines.append("| " + " | ".join([
                str(row["candidate_method"]), str(row["model_id"]), str(row["paired_item_count"]),
                _format(row["clean_target_success_rate"]), _format(row["adversarial_target_success_rate"]),
                _format(row["conditioned_asr"]), _format(row["source_suppression_rate"]),
                _format(row["refusal_rate"]), _format(row["api_failure_rate"]),
            ]) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tables from frozen API replay records without tuning.")
    parser.add_argument("--result-dir", action="append", required=True, dest="result_dirs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    output = Path(args.output)
    records = load_records([Path(value) for value in args.result_dirs])
    overview = rubric_rows(records, args.bootstrap_samples)
    items = item_rows(records)
    _atomic_csv(output / "api_rubric_overview.csv", overview, RUBRIC_FIELDS)
    for task in sorted({row["task_type"] for row in overview}):
        _atomic_csv(output / f"api_rubric_{task}.csv", [row for row in overview if row["task_type"] == task], RUBRIC_FIELDS)
    _atomic_csv(output / "api_paired_item_outcomes.csv", items, list(items[0]) if items else ["task_type"])
    write_markdown(output / "api_rubric.md", overview)
    print(json.dumps({"records": len(records), "rubric_rows": len(overview), "paired_rows": len(items), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
