from __future__ import annotations

import argparse
import json
from pathlib import Path

from faireval.execute import execute_plan


LOCAL_FAMILIES = {"qwen25_local", "phi35_local"}


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
                    "instruction": "Add --execute and --code-commit-sha after reviewing this batch.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.code_commit_sha:
        raise ValueError("--code-commit-sha is required with --execute")

    summary = execute_plan(
        plan_dir=Path(args.plan_dir),
        freeze_root=Path(args.freeze_root),
        output_jsonl=Path(args.output_jsonl),
        code_commit_sha=args.code_commit_sha,
        families={args.family},
        max_cells=args.max_cells,
    )
    summary["whitebox_family_sequential_execution"] = True
    summary["family"] = args.family
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
