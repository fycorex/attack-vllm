#!/usr/bin/env python3
"""Aggregate final replay into TASR, selector, and M1--M6 tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from proxy_selector.io import atomic_json


def main() -> None:
    output = Path("outputs/proxy_selector_pilot")
    cka = json.loads((output / "cka" / "cka_bootstrap.json").read_text())["seed42"]
    rows = []
    for target in ("T1", "T2"):
        rows.extend(json.loads((output / "vllm" / target / "test_replay.json").read_text())["rows"])
    pair_rows, cells = [], defaultdict(list)
    for row in rows:
        valid = attacked = random = 0
        for question in row["questions"]:
            qid = str(question["question_id"]); target_answer = question["answer"]
            if row["controls"][qid]["target"]["normalized_answer"] == target_answer:
                valid += 1
                attacked += row["adversarial"][qid]["normalized_answer"] == target_answer
                random += row["controls"][qid]["random"]["normalized_answer"] == target_answer
        if not valid: continue
        pair = {"pair_id": row["pair_id"], "proxy": row["proxy"], "target": row["target"], "recipe": row["recipe"], "pair_tasr": attacked / valid, "pair_random_tasr": random / valid}
        pair_rows.append(pair); cells[(row["proxy"], row["target"])].append(pair)
    matrix = []
    for (proxy, target), values in sorted(cells.items()):
        tasr = sum(item["pair_tasr"] for item in values) / len(values)
        random = sum(item["pair_random_tasr"] for item in values) / len(values)
        key = f"{proxy}->{target}"; interval = cka[key]
        matrix.append({"pair_id": {("P1", "T1"): "M1", ("P1", "T2"): "M2", ("P2", "T1"): "M3", ("P2", "T2"): "M4", ("P3", "T1"): "M5", ("P3", "T2"): "M6"}[(proxy,target)], "proxy": proxy, "proxy_type": "VLM" if proxy == "P1" else "Encoder", "target": target, "cross_family": True, "cka": interval["cka"], "cka_ci_low": interval["ci_low"], "cka_ci_high": interval["ci_high"], "tasr": tasr, "random_tasr": random, "delta_tasr": tasr-random, "selected_recipe": values[0]["recipe"], "n_pairs": len(values)})
    summaries = output / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    with (summaries / "pair_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(pair_rows[0]));writer.writeheader();writer.writerows(pair_rows)
    with (output / "summaries" / "proxy_target_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(matrix[0]));writer.writeheader();writer.writerows(matrix)
    selector = {}
    for target in ("T1", "T2"):
        target_rows = [row for row in matrix if row["target"] == target]
        oracle = max(target_rows, key=lambda row: row["tasr"])
        selected = max(target_rows, key=lambda row: row["cka"])
        selector[target] = {"oracle_proxy": oracle["proxy"], "oracle_tasr": oracle["tasr"], "cka_selected_proxy": selected["proxy"], "cka_selected_tasr": selected["tasr"], "regret": oracle["tasr"] - selected["tasr"], "top2": selected["proxy"] in [row["proxy"] for row in sorted(target_rows,key=lambda row: row["tasr"],reverse=True)[:2]]}
    atomic_json(output / "summaries" / "selector_results.json", selector)
    print(json.dumps({"matrix": matrix, "selector": selector}, indent=2))


if __name__ == "__main__": main()
