from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen25_local", "phi35_local")


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _run_family(
    *,
    family: str,
    plan_dir: Path,
    freeze_root: Path,
    output_jsonl: Path,
    preexecution_seal: Path,
    code_commit_sha: str,
    max_cells: int,
    execute: bool,
    max_process_retries: int,
    retry_cooldown_seconds: int,
) -> None:
    args = [
        sys.executable,
        str(ROOT / "scripts" / "run_whitebox_family.py"),
        "--plan-dir",
        str(plan_dir),
        "--freeze-root",
        str(freeze_root),
        "--family",
        family,
        "--output-jsonl",
        str(output_jsonl),
        "--max-cells",
        str(max_cells),
        "--preexecution-seal",
        str(preexecution_seal),
    ]
    if execute:
        args.extend(["--code-commit-sha", code_commit_sha, "--execute"])

    print("\n" + "=" * 88, flush=True)
    print(f"FAIREVAL CANONICAL WHITE-BOX FAMILY: {family}", flush=True)
    print("=" * 88, flush=True)

    # Native CUDA libraries such as bitsandbytes can terminate the child process
    # without raising a Python exception. Retrying the family process is safe
    # because execute_plan resumes strictly by persisted planned_cell_id values:
    # completed experimental outcomes are never regenerated, while the interrupted
    # cell has no persisted row and remains pending.
    attempts = max_process_retries + 1 if execute else 1
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(args, cwd=ROOT)
        if completed.returncode == 0:
            return
        if attempt >= attempts:
            break
        print(
            f"{family}: child process exited {completed.returncode}; "
            f"cooling down {retry_cooldown_seconds}s before exact-resume retry "
            f"{attempt + 1}/{attempts}.",
            flush=True,
        )
        time.sleep(retry_cooldown_seconds)

    raise SystemExit(
        f"canonical white-box execution stopped for {family} "
        f"with exit code {completed.returncode} after {attempts} attempt(s). "
        "Completed planned_cell_id rows are preserved; diagnose the native runtime "
        "before resuming."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical two-model FairEval white-box campaign sequentially. "
            "This wrapper uses the immutable plan and exact resume semantics; it never "
            "uses the retired generic-prompt smoke outputs."
        )
    )
    parser.add_argument(
        "--plan-dir",
        default="results/plans/whitebox-full-v1/core",
    )
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument(
        "--output-dir",
        default="results/runs/whitebox-full-v2",
    )
    parser.add_argument(
        "--preexecution-seal",
        default="results/preexecution/seal-v5/PREEXECUTION_SEAL.json",
    )
    parser.add_argument(
        "--max-cells-per-family",
        type=int,
        default=6480,
        help="6480 completes one family for the 360x6x3 FairSynth core plan.",
    )
    parser.add_argument(
        "--max-process-retries",
        type=int,
        default=3,
        help=(
            "Retry a family child process after native-runtime termination. "
            "Exact planned_cell_id resume prevents re-running persisted cells."
        ),
    )
    parser.add_argument(
        "--retry-cooldown-seconds",
        type=int,
        default=45,
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.max_cells_per_family <= 0:
        raise ValueError("--max-cells-per-family must be positive")
    if args.max_process_retries < 0:
        raise ValueError("--max-process-retries must be non-negative")
    if args.retry_cooldown_seconds < 0:
        raise ValueError("--retry-cooldown-seconds must be non-negative")

    head = _git_head()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "faireval-whitebox-full-wrapper-v1",
        "git_commit_sha": head,
        "plan_dir": args.plan_dir,
        "freeze_root": args.freeze_root,
        "preexecution_seal": args.preexecution_seal,
        "families": list(FAMILIES),
        "max_cells_per_family": args.max_cells_per_family,
        "max_process_retries": args.max_process_retries,
        "retry_cooldown_seconds": args.retry_cooldown_seconds,
        "execute": bool(args.execute),
        "retired_smoke_outputs_used": False,
    }
    (output_dir / "wrapper_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for family in FAMILIES:
        _run_family(
            family=family,
            plan_dir=Path(args.plan_dir),
            freeze_root=Path(args.freeze_root),
            output_jsonl=output_dir / f"{family}.jsonl",
            preexecution_seal=Path(args.preexecution_seal),
            code_commit_sha=head,
            max_cells=args.max_cells_per_family,
            execute=args.execute,
            max_process_retries=args.max_process_retries,
            retry_cooldown_seconds=args.retry_cooldown_seconds,
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": "execute" if args.execute else "dry_run",
                "git_commit_sha": head,
                "output_dir": str(output_dir),
                "families": list(FAMILIES),
                "next": (
                    "Run scripts/finalize_whitebox.py with both family JSONL files "
                    "after complete coverage."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
