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
EXPECTED_CELLS_PER_FAMILY = 72
EXPECTED_TOTAL = 144


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
        "--plan-dir",
        str(plan_dir),
        "--freeze-root",
        "data/frozen",
        "--family",
        family,
        "--output-jsonl",
        str(output_jsonl),
        "--max-cells",
        str(EXPECTED_CELLS_PER_FAMILY),
        "--preexecution-seal",
        str(seal),
        "--plan-key",
        "whitebox_v6_canary",
        "--code-commit-sha",
        head,
        "--execute",
    ]
    for attempt in range(retries + 1):
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode == 0:
            return
        if attempt >= retries:
            raise SystemExit(
                f"V6 canary family {family} stopped with exit code {completed.returncode}; "
                "persisted planned_cell_id rows remain resumable and must not be regenerated."
            )
        print(
            f"{family}: child exited {completed.returncode}; exact-resume retry "
            f"{attempt + 2}/{retries + 1} after {cooldown}s",
            flush=True,
        )
        time.sleep(cooldown)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute only the sealed 144-cell FairEval V6 protocol canary."
    )
    parser.add_argument("--plan-dir", default="results/plans/whitebox-v6-canary/core")
    parser.add_argument(
        "--output-dir",
        default="results/runs/whitebox-v6-canary",
    )
    parser.add_argument(
        "--preexecution-seal",
        default="results/preexecution/seal-v11/PREEXECUTION_SEAL.json",
    )
    parser.add_argument("--max-process-retries", type=int, default=2)
    parser.add_argument("--retry-cooldown-seconds", type=int, default=30)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        raise ValueError("V6 canary wrapper requires explicit --execute")
    if args.max_process_retries < 0 or args.retry_cooldown_seconds < 0:
        raise ValueError("retry controls must be non-negative")

    output_dir = Path(args.output_dir)
    normalized = output_dir.as_posix().rstrip("/").lower()
    if normalized.endswith("results/runs/whitebox-full-v4"):
        raise ValueError("refusing to write V6 data into the frozen V4 output directory")

    plan_dir = Path(args.plan_dir)
    cells, manifest = load_and_verify_plan(plan_dir)
    if manifest.get("schema_version") != "faireval-whitebox-v6-canary-plan-v1":
        raise ValueError("not a V6 canary plan")
    if len(cells) != EXPECTED_TOTAL:
        raise ValueError(f"V6 canary must contain exactly {EXPECTED_TOTAL} cells")
    family_counts = Counter(str(row["model_family"]) for row in cells)
    if family_counts != Counter({family: EXPECTED_CELLS_PER_FAMILY for family in FAMILIES}):
        raise ValueError(f"V6 canary family geometry drift: {dict(family_counts)}")

    head = _git_head()
    seal = Path(args.preexecution_seal)
    verification = verify_preexecution_seal(
        seal,
        expected_commit_sha=head,
        plan_dir=plan_dir,
        plan_key="whitebox_v6_canary",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper_manifest = {
        "schema_version": "faireval-whitebox-v6-canary-execution-v1",
        "git_commit_sha": head,
        "plan_sha256": manifest.get("plan_sha256"),
        "run_plan_file_sha256": manifest.get("run_plan_file_sha256"),
        "preexecution_seal": str(seal),
        "preexecution_seal_verification": verification,
        "families": list(FAMILIES),
        "cells_per_family": EXPECTED_CELLS_PER_FAMILY,
        "planned_cells": EXPECTED_TOTAL,
        "generative_format_repair": False,
        "frozen_v4_output_used_as_destination": False,
    }
    (output_dir / "wrapper_manifest.json").write_text(
        json.dumps(wrapper_manifest, indent=2, sort_keys=True) + "\n",
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

    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "git_commit_sha": head,
                "output_dir": str(output_dir),
                "planned_cells": EXPECTED_TOTAL,
                "instruction": (
                    "Do not scale. Run scripts/audit_whitebox_v6_canary.py and obey "
                    "the predeclared per-model >=0.95 semantic-validity gate."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
