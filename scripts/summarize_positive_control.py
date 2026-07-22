#!/usr/bin/env python3
"""Aggregate strict, controlled, and raw transfer outcomes from replay JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from proxy_selector.io import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/proxy_selector_pilot"))
    args = parser.parse_args()
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for file in args.output.glob("positive_control/vllm/T*/*.json"):
        target = file.parent.name
        for row in json.loads(file.read_text()).get("rows", []):
            key = (row["proxy"], target)
            groups[key]["n"] += 1
            groups[key]["strict_targeted"] += int(bool(row.get("strict_targeted_success", row.get("success", False))))
            groups[key]["controlled_change"] += int(bool(row.get("answer_change_transfer", False)))
            groups[key]["raw_change"] += int(bool(row.get("raw_answer_change", False)))
    rows = []
    for (proxy, target), counts in sorted(groups.items()):
        n = counts["n"]
        rows.append({
            "proxy": proxy, "target": target, "images_replayed": n,
            "strict_targeted_hits": counts["strict_targeted"],
            "strict_targeted_rate": counts["strict_targeted"] / n if n else None,
            "controlled_change_hits": counts["controlled_change"],
            "controlled_change_rate": counts["controlled_change"] / n if n else None,
            "raw_change_hits": counts["raw_change"],
            "raw_change_rate": counts["raw_change"] / n if n else None,
        })
    output = args.output / "positive_control" / "summaries" / "transfer_rates.json"
    atomic_json(output, {
        "rows": rows,
        "cka_files": [
            "outputs/proxy_selector_pilot/cka/cka_seed42.csv",
            "outputs/proxy_selector_pilot/cka/cka_seed43.csv",
            "outputs/proxy_selector_pilot/cka/cka_bootstrap.json",
        ],
    })
    print(json.dumps({"rows": rows, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
