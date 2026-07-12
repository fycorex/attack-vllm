#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transfer_eval import run_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay existing clean/adversarial pairs on multimodal APIs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", action="append", required=True, dest="output_dirs")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-real-api", action="store_true", help="Explicitly opt in to billable network requests.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-estimated-cost", type=float)
    args = parser.parse_args()
    result = run_replay(Path(args.config), [Path(x) for x in args.output_dirs], Path(args.result_dir),
        allow_real_api=args.allow_real_api, dry_run=args.dry_run, limit=args.limit, resume=args.resume,
        max_requests=args.max_requests, max_estimated_cost=args.max_estimated_cost)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
