#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import yaml


def call(command: list[str]) -> None:
    subprocess.run(command, check=True)


def analyze(python: str, root: Path, stage: str) -> list[dict]:
    output = root / stage / "aggregate.json"
    call([python, "scripts/analyze_noise_search.py", "--root", str(root), "--stage", stage, "--output", str(output)])
    return json.loads(output.read_text())


def candidates(rows: list[dict], count: int) -> list[dict]:
    selected = []; seen = set()
    ranked = sorted((r for r in rows if r["mode"] != "none"), key=lambda r: (r["paired_delta"] if r["paired_delta"] is not None else -2,
                                                                                 r["paired_margin_delta"] if r.get("paired_margin_delta") is not None else -2,
                                                                                 r["mean_heldout_macro_asr"]), reverse=True)
    for row in ranked:
        key = (row["mode"], row["sigma"], row["samples"])
        if key in seen: continue
        seen.add(key); selected.append({"mode": key[0], "sigma": key[1], "samples": key[2]})
        if len(selected) == count: break
    return selected


def write_promoted(base: dict, source_stage: str, target_stage: str, selected: list[dict], path: Path) -> None:
    spec = deepcopy(base); stage = spec["stages"][target_stage]
    modes = sorted({x["mode"] for x in selected} | {"none"}); sigmas = sorted({float(x["sigma"]) for x in selected} | {0.0}); samples = sorted({int(x["samples"]) for x in selected} | {1})
    stage["noise"] = {"modes": modes, "sigmas": sigmas, "samples": samples}
    stage["promoted_from"] = source_stage; stage["promoted_candidates"] = selected
    path.write_text(yaml.safe_dump(spec, sort_keys=False))


def run_stage(python: str, spec: Path, stage: str, root: Path, workers: int) -> None:
    call([python, "scripts/run_noise_search.py", "--spec", str(spec), "--stage", stage, "--root", str(root), "--python", python, "--workers", str(workers)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Successive-halving noise search with resumable promotion.")
    parser.add_argument("--spec", default="configs/noise_search.yaml"); parser.add_argument("--root", default="outputs/noise_search")
    parser.add_argument("--until", choices=["stage0", "stage1", "equal_forward", "final"], default="final")
    parser.add_argument("--python", default=sys.executable); parser.add_argument("--stage0-top-k", type=int, default=6); parser.add_argument("--stage1-top-k", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(); root = Path(args.root); root.mkdir(parents=True, exist_ok=True); base_path = Path(args.spec); base = yaml.safe_load(base_path.read_text())
    run_stage(args.python, base_path, "stage0", root, args.workers); rows0 = analyze(args.python, root, "stage0")
    promoted1 = root / "promoted_stage1.yaml"; write_promoted(base, "stage0", "stage1", candidates(rows0, args.stage0_top_k), promoted1)
    if args.until == "stage0": return
    run_stage(args.python, promoted1, "stage1", root, args.workers); rows1 = analyze(args.python, root, "stage1")
    if args.until == "stage1": return
    run_stage(args.python, base_path, "equal_forward", root, args.workers); analyze(args.python, root, "equal_forward")
    if args.until == "equal_forward": return
    promoted_final = root / "promoted_final.yaml"; write_promoted(base, "stage1", "final", candidates(rows1, args.stage1_top_k), promoted_final)
    run_stage(args.python, promoted_final, "final", root, args.workers); analyze(args.python, root, "final")


if __name__ == "__main__": main()
