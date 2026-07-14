#!/usr/bin/env python3
"""Run a follow-up command only after a transfer-search cycle completes cleanly.

The launcher deliberately treats a soft-deadline exit as a failure gate: a
large confirmation must use frozen, fully replicated candidates rather than
whatever happened to finish before a time limit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


SUCCESS = {"complete"}
STOPPED = {"screen_complete_deadline_reached", "failed", "error"}


def write_status(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"updated_at": datetime.now(timezone.utc).isoformat(), **values}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for a completed transfer-search cycle, then run a separately configured confirmation."
    )
    parser.add_argument("--cycle-root", required=True, help="Output directory containing cycle_state.json.")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--log", help="Follow-up stdout/stderr log; defaults under cycle root.")
    parser.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        required=True,
        help="Command to run after '--command'. It is executed without a shell.",
    )
    args = parser.parse_args()
    if not args.command:
        parser.error("provide a follow-up command after --command")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    root = Path(args.cycle_root).resolve()
    state_path = root / "cycle_state.json"
    status_path = root / "post_cycle_launcher_state.json"
    log_path = Path(args.log).resolve() if args.log else root / "post_cycle_confirmation.log"
    write_status(status_path, status="waiting", cycle_root=str(root), command=args.command)

    while True:
        if not state_path.is_file():
            write_status(status_path, status="waiting_for_state", cycle_root=str(root), command=args.command)
            time.sleep(args.poll_seconds)
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        status = str(state.get("status", "unknown"))
        if status in SUCCESS:
            required = [root / "final_selection.json", root / "augmentation_final_selection.json"]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                write_status(status_path, status="blocked_missing_frozen_selection", missing=missing, cycle_state=state)
                raise SystemExit("Cycle reported complete without frozen selections; refusing follow-up.")
            break
        if status in STOPPED:
            write_status(status_path, status="blocked_cycle_not_complete", cycle_state=state)
            raise SystemExit(f"Cycle ended with status '{status}'; refusing follow-up.")
        write_status(status_path, status="waiting", cycle_state=state, command=args.command)
        time.sleep(args.poll_seconds)

    write_status(status_path, status="starting_follow_up", cycle_state=state, command=args.command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(args.command, stdout=log, stderr=subprocess.STDOUT, check=False)
    final = "complete" if result.returncode == 0 else "follow_up_failed"
    write_status(status_path, status=final, returncode=result.returncode, command=args.command, log=str(log_path))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
