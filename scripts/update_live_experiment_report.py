#!/usr/bin/env python3
"""Render a local, Git-ignored status report from experiment artifacts."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = ROOT.parent / "attack-vllm-alignment"


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def trial_counts(path: Path) -> tuple[int, int]:
    rows = [read_json(value, {}) for value in path.glob("*/trial_result.json")]
    return sum(row.get("status") == "complete" for row in rows), sum(row.get("status") == "failed" for row in rows)


def paired_history() -> list[str]:
    path = ROOT / "outputs/surrogate_cross_dataset_equal_forwards/analysis/aggregate_paired_comparisons.csv"
    if not path.is_file():
        return ["- Historical equal-forward table is unavailable."]
    result = []
    for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
        if row.get("scope") == "macro_targets" and row.get("baseline") == "single_reference" and row.get("candidate") in {"two_homogeneous", "four_lightweight_mixed"}:
            result.append(
                f"- {row['dataset']}: `{row['candidate']}` vs single = {100 * float(row['asr_delta']):+.1f} pp "
                f"(95% paired CI [{100 * float(row['ci95_low']):+.1f}, {100 * float(row['ci95_high']):+.1f}] pp)."
            )
    return result or ["- No historical paired macro comparisons found."]


def alignment_history() -> list[str]:
    root = ALIGNMENT / "outputs/alignment_analysis/stage1_clustered"
    result = []
    path = root / "cka_transfer_analysis.csv"
    if path.is_file():
        for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
            if row.get("predictor") in {"centered_linear_cka", "neighborhood_overlap_at_1"}:
                result.append(f"- {row['predictor']} → held-out ASR: Spearman {float(row['spearman']):.3f}, CI {row['ci95']}.")
    distance = read_json(root / "distance_summary.json", {})
    for value in distance.get("correlations", []):
        if value.get("predictor") == "delta_dtarget" and value.get("outcome") == "target_success":
            result.append(f"- ΔD_target → success: Spearman {float(value['spearman']):.3f}, CI {value['ci95']}.")
    return result or ["- No completed alignment summary found."]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-root", default=str(ROOT / "outputs/transfer_search_cycle"))
    parser.add_argument("--output")
    args = parser.parse_args()
    cycle = Path(args.cycle_root)
    output = Path(args.output) if args.output else cycle / "LIVE_EXPERIMENT_REPORT.md"
    state = read_json(cycle / "cycle_state.json", {})
    smoke = read_json(cycle / "api_capability_smoke.json", {})
    surrogate_complete, surrogate_failed = trial_counts(cycle / "attacks/cross_family_screen")
    augmentation_complete, augmentation_failed = trial_counts(cycle / "augmentation/strict_equal_forward_screen")
    measurements = []
    for dataset in ("caption_caltech", "llava_vqa", "receipt_ocr"):
        manifest = read_json(cycle / "alignment" / dataset / "run_manifest.json", {})
        if manifest:
            measurements.append(f"- {dataset}: {len(manifest.get('item_ids', []))} images; {len(manifest.get('heldout_models', []))} held-out models.")
    api = next(iter(smoke.get("results", [])), {})
    lines = [
        "# Live Transferability Experiment Report", "",
        f"Refreshed: {datetime.now(timezone.utc).isoformat()}", "",
        "## Current cycle", "",
        f"- State: `{state.get('status', 'unknown')}`.",
        f"- Updated by runner: `{state.get('updated_at', 'unknown')}`.",
        f"- Cross-family surrogate screening: {surrogate_complete}/30 complete; {surrogate_failed} failed.",
        f"- Equal-forward augmentation screening: {augmentation_complete}/27 complete; {augmentation_failed} failed.",
        "- API outcomes are excluded from all search and promotion decisions.", "",
        "## Completed measurements in the active cycle", "",
        *(measurements or ["- Measurement files have not been written yet."]), "",
        "## API capability check", "",
        f"- requested: `{api.get('model_id', 'not checked')}`; resolved: `{api.get('resolved_model_id', 'not available')}`; vision request: `{api.get('vision_request_ok', False)}`.", "",
        "## Previously completed equal-forward evidence", "", *paired_history(), "",
        "## Previously completed alignment evidence", "", *alignment_history(), "",
        "## Interpretation status", "",
        "- The historical two-proxy improvement is a completed held-out encoder result, not yet an API conclusion.",
        "- Global CKA and local-neighborhood findings remain model-family and dataset dependent; the active cross-family run tests that limitation.",
        "- No current-cycle attack result is reported until its trial record is complete and validation has passed.", "",
        "## Next automatic milestones", "",
        "1. Complete screening and check equal-forward accounting.",
        "2. Promote candidates using held-out open-source scores only.",
        "3. Replicate promoted candidates with two additional seeds.",
        "4. Freeze candidates, run API replay, and render task-specific rubric tables.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
