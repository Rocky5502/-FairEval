from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from faireval.execute import load_and_verify_plan
from faireval.preexecution import verify_preexecution_seal


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen25_local", "phi35_local")
EXPECTED_CELLS_PER_FAMILY = 1440
EXPECTED_TOTAL = 2880


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _run_family(
    *,
    family: str,
    plan_dir: Path,
    output_jsonl: Path,
    seal: Path,
    head: str,
    retries: int,
    cooldown: int,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_whitebox_family.py"),
        "--plan-dir", str(plan_dir),
        "--freeze-root", "data/frozen",
        "--family", family,
        "--output-jsonl", str(output_jsonl),
        "--max-cells", str(EXPECTED_CELLS_PER_FAMILY),
        "--preexecution-seal", str(seal),
        "--plan-key", "whitebox_v7_lean",
        "--code-commit-sha", head,
        "--execute",
    ]
    for attempt in range(retries + 1):
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode == 0:
            return
        if attempt >= retries:
            raise SystemExit(
                f"V7 lean family {family} stopped with exit code {completed.returncode}; "
                "persisted planned_cell_id rows remain resumable."
            )
        print(
            f"{family}: child exited {completed.returncode}; exact-resume retry "
            f"{attempt + 2}/{retries + 1} after {cooldown}s",
            flush=True,
        )
        time.sleep(cooldown)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute the sealed complete 2,880-cell FairEval V7 lean local run."
    )
    parser.add_argument("--plan-dir", default="results/plans/whitebox-v7-lean/core")
    parser.add_argument("--output-dir", default="results/runs/whitebox-v7-lean")
    parser.add_argument(
        "--preexecution-seal",
        default="results/preexecution/seal-v13/PREEXECUTION_SEAL.json",
    )
    parser.add_argument("--max-process-retries", type=int, default=2)
    parser.add_argument("--retry-cooldown-seconds", type=int, default=30)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        raise ValueError("V7 lean runner requires explicit --execute")
    if args.max_process_retries < 0 or args.retry_cooldown_seconds < 0:
        raise ValueError("retry controls must be non-negative")

    plan_dir = Path(args.plan_dir)
    cells, manifest = load_and_verify_plan(plan_dir)
    if manifest.get("schema_version") != "faireval-whitebox-v7-lean-plan-v1":
        raise ValueError("not the V7 lean main plan")
    if len(cells) != EXPECTED_TOTAL:
        raise ValueError(f"V7 lean plan must contain exactly {EXPECTED_TOTAL} cells")
    family_counts = Counter(str(row["model_family"]) for row in cells)
    expected = Counter({family: EXPECTED_CELLS_PER_FAMILY for family in FAMILIES})
    if family_counts != expected:
        raise ValueError(f"V7 lean family geometry drift: {dict(family_counts)}")

    head = _git_head()
    seal = Path(args.preexecution_seal)
    verification = verify_preexecution_seal(
        seal,
        expected_commit_sha=head,
        plan_dir=plan_dir,
        plan_key="whitebox_v7_lean",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = {
        "schema_version": "faireval-whitebox-v7-lean-execution-v1",
        "git_commit_sha": head,
        "plan_sha256": manifest.get("plan_sha256"),
        "run_plan_file_sha256": manifest.get("run_plan_file_sha256"),
        "preexecution_seal": str(seal),
        "preexecution_seal_verification": verification,
        "families": list(FAMILIES),
        "cells_per_family": EXPECTED_CELLS_PER_FAMILY,
        "planned_cells": EXPECTED_TOTAL,
        "complete_matrix_required": True,
        "generative_format_repair": False,
        "prompt_interface_version": "faireval-prompt-interface-v7",
    }
    (output_dir / "wrapper_manifest.json").write_text(
        json.dumps(wrapper, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for family in FAMILIES:
        _run_family(
            family=family,
            plan_dir=plan_dir,
            output_jsonl=output_dir / f"{family}.jsonl",
            seal=seal,
            head=head,
            retries=args.max_process_retries,
            cooldown=args.retry_cooldown_seconds,
        )

    print(json.dumps({
        "status": "COMPLETE",
        "git_commit_sha": head,
        "output_dir": str(output_dir),
        "planned_cells": EXPECTED_TOTAL,
        "instruction": "Audit the complete V7 lean run before any paper-facing analysis.",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
