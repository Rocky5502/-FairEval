from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from faireval.budget import BudgetExceeded, ZhizengzengBudgetGuard
from faireval.execute import completed_cell_ids, execute_plan, load_and_verify_plan, pending_cells


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a frozen FairEval hosted plan through Zhizengzeng with a "
            "balance-reconciled RMB hard cap. Dry-run by default."
        )
    )
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--freeze-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--ledger", default="results/budget/hosted_zzz_v1.json")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--family", action="append")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--target-rmb", type=float, default=200.0)
    parser.add_argument("--hard-cap-rmb", type=float, default=250.0)
    parser.add_argument("--request-reserve-rmb", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    _load_dotenv(Path(args.env_file))
    os.environ["FAIREVAL_HOSTED_GATEWAY"] = "zhizengzeng"
    os.environ.setdefault("ZZZ_BASE_URL", "https://api.zhizengzeng.com/v1")

    plan_dir = Path(args.plan_dir)
    output_jsonl = Path(args.output_jsonl)
    families = set(args.family) if args.family else None
    cells, manifest = load_and_verify_plan(plan_dir)
    completed = completed_cell_ids(output_jsonl)
    selected = pending_cells(cells, completed=completed, families=families)
    if args.max_cells is not None:
        if args.max_cells <= 0:
            raise ValueError("--max-cells must be positive")
        selected = selected[: args.max_cells]

    dry_report = {
        "schema_version": "faireval-hosted-budgeted-launch-v1",
        "mode": "execute" if args.execute else "dry_run",
        "gateway": "zhizengzeng",
        "plan_sha256": manifest.get("plan_sha256"),
        "planned_cells_total": len(cells),
        "completed_cells": len(completed),
        "selected_pending_cells": len(selected),
        "family_filter": None if families is None else sorted(families),
        "target_rmb": args.target_rmb,
        "hard_cap_rmb": args.hard_cap_rmb,
        "request_reserve_rmb": args.request_reserve_rmb,
        "api_calls_made": 0,
    }
    if not args.execute:
        print(json.dumps(dry_report, indent=2, sort_keys=True))
        return 0

    if not os.environ.get("ZZZ_API_KEY"):
        raise RuntimeError("ZZZ_API_KEY is required for hosted execution")

    guard = ZhizengzengBudgetGuard(
        ledger_path=Path(args.ledger),
        target_rmb=args.target_rmb,
        hard_cap_rmb=args.hard_cap_rmb,
        request_reserve_rmb=args.request_reserve_rmb,
    )
    initial = guard.ensure_initialized()
    if initial.spent_rmb >= args.hard_cap_rmb:
        raise BudgetExceeded(
            f"Refusing to start: existing experiment spend {initial.spent_rmb:.4f} RMB "
            f"already reaches hard cap {args.hard_cap_rmb:.2f} RMB"
        )

    try:
        summary = execute_plan(
            plan_dir=plan_dir,
            freeze_root=Path(args.freeze_root),
            output_jsonl=output_jsonl,
            code_commit_sha=_git_head(),
            families=families,
            max_cells=args.max_cells,
            before_cell=guard.before_cell,
            after_cell=guard.after_cell,
        )
    except BudgetExceeded as exc:
        report = guard.report()
        report["execution_status"] = "STOPPED_BY_BUDGET_GUARD"
        report["reason"] = str(exc)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 10

    report = guard.report()
    summary["budget"] = report
    summary["execution_status"] = "COMPLETED_SELECTED_CELLS"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
