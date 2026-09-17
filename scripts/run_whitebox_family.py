from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from faireval.execute import execute_plan
from faireval.preexecution import verify_preexecution_seal


LOCAL_FAMILIES = {"qwen25_local", "phi35_local"}
ROOT = Path(__file__).resolve().parents[1]


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "real white-box execution requires a Git checkout so code provenance can be proven"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one FairEval white-box family at a time with exact resume semantics. "
            "Dry-run by default; pass --execute for real GPU inference."
        )
    )
    parser.add_argument("--plan-dir", default="results/plans/whitebox-full-v1/core")
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument("--family", required=True, choices=sorted(LOCAL_FAMILIES))
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-cells", type=int, default=250)
    parser.add_argument("--code-commit-sha")
    parser.add_argument(
        "--preexecution-seal",
        default="results/preexecution/seal-v1/PREEXECUTION_SEAL.json",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.max_cells <= 0:
        raise ValueError("--max-cells must be positive")

    if not args.execute:
        from faireval.execute import completed_cell_ids, load_and_verify_plan, pending_cells

        cells, manifest = load_and_verify_plan(Path(args.plan_dir))
        completed = completed_cell_ids(Path(args.output_jsonl))
        selected = pending_cells(cells, completed=completed, families={args.family})
        selected = selected[: args.max_cells]
        seal_path = Path(args.preexecution_seal)
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "family": args.family,
                    "plan_sha256": manifest.get("plan_sha256"),
                    "completed_cells": len(completed),
                    "selected_pending_cells": len(selected),
                    "max_cells": args.max_cells,
                    "model_weights_loaded": False,
                    "hosted_api_calls_made": 0,
                    "preexecution_seal_path": str(seal_path),
                    "preexecution_seal_exists": seal_path.is_file(),
                    "instruction": (
                        "Build/verify the zero-call pre-execution seal, then add --execute "
                        "and --code-commit-sha after reviewing this batch."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.code_commit_sha:
        raise ValueError("--code-commit-sha is required with --execute")
    checked_out_sha = _git_head()
    if args.code_commit_sha.strip() != checked_out_sha:
        raise ValueError(
            "--code-commit-sha must equal the currently checked-out Git HEAD; "
            f"argument={args.code_commit_sha.strip()} HEAD={checked_out_sha}. "
            "Do not mix code revisions inside a frozen run log."
        )

    seal_verification = verify_preexecution_seal(
        Path(args.preexecution_seal),
        expected_commit_sha=checked_out_sha,
        plan_dir=Path(args.plan_dir),
        plan_key="whitebox_core",
    )

    summary = execute_plan(
        plan_dir=Path(args.plan_dir),
        freeze_root=Path(args.freeze_root),
        output_jsonl=Path(args.output_jsonl),
        code_commit_sha=checked_out_sha,
        families={args.family},
        max_cells=args.max_cells,
    )
    summary["whitebox_family_sequential_execution"] = True
    summary["family"] = args.family
    summary["checked_out_code_commit_sha"] = checked_out_sha
    summary["preexecution_seal"] = seal_verification
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
