# Read the chosen pair, model side and phase from command-line arguments.
# Inspect the plan by default; the explicit execution option calls the shared worker.
# Saved attempt records distinguish execution from later acceptance.

"""Inspect or run one worker using its own Python environment."""

# Keep disposable interpreter files out of the submitted source folder.
import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a1.worker import run_worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=[f"P0{p}_{s}" for p in (3, 2, 1, 4) for s in "AB"], required=True)
    parser.add_argument("--phase", choices=["pilot", "final", "reproduction"], default="pilot")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    pair, side = args.worker.split("_")
    result = run_worker(ROOT, pair, side, args.phase, args.execute, backup_root=args.backup_root)
    print(json.dumps(result, indent=2))
    if args.execute and any(row["status"] not in {"success", "resumed"} for row in result):
        raise SystemExit(1)
