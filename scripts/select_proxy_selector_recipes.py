#!/usr/bin/env python3
"""Freeze one development-selected recipe per proxy using pair-macro TASR."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from proxy_selector.io import atomic_json


def main() -> None:
    output = Path("outputs/proxy_selector_pilot")
    rows = []
    for target in ("T1", "T2"):
        rows.extend(json.loads((output / "vllm" / target / "dev_replay.json").read_text())["rows"])
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_target: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        successes, valid = 0, 0
        for question in row["questions"]:
            qid = str(question["question_id"])
            if row["controls"][qid]["target"]["normalized_answer"] == question["answer"]:
                valid += 1
                successes += row["adversarial"][qid]["normalized_answer"] == question["answer"]
        if valid:
            value = successes / valid
            scores[(row["proxy"], row["recipe"])].append(value)
            per_target[(row["proxy"], row["recipe"], row["target"])].append(value)
    selected = {}
    for proxy in ("P1", "P2", "P3"):
        candidates = []
        for recipe in ("C0", "C1", "C2"):
            values = scores[(proxy, recipe)]
            mean = sum(values) / len(values) if values else float("-inf")
            target_means = [sum(per_target[(proxy, recipe, target)]) / len(per_target[(proxy, recipe, target)]) for target in ("T1", "T2") if per_target[(proxy, recipe, target)]]
            std = (sum((x - mean) ** 2 for x in target_means) / len(target_means)) ** .5 if target_means else float("inf")
            candidates.append((recipe, mean, std))
        best_mean = max(item[1] for item in candidates)
        tied = [item for item in candidates if best_mean - item[1] <= .005]
        chosen = sorted(tied, key=lambda item: (item[2], ("C0", "C1", "C2").index(item[0])))[0]
        selected[proxy] = {"recipe": chosen[0], "mean_tasr": chosen[1], "target_std": chosen[2], "all_recipes": [{"recipe": x[0], "mean_tasr": x[1], "target_std": x[2]} for x in candidates]}
    atomic_json(output / "dev" / "selected_recipes.json", selected)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
