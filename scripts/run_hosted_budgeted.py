from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from faireval.budget import BudgetExceeded, ZhizengzengBudgetGuard
from faireval.execute import completed_cell_ids, execute_plan, load_and_verify_plan, pending_cells
from faireval.gateway import verify_frozen_gateway_models
from faireval.preexecution import verify_preexecution_seal


FROZEN_MAX_TARGET_RMB = 200.0
FROZEN_MAX_EMERGENCY_THRESHOLD_RMB = 250.0
ROOT = Path(__file__).resolve().parents[1]


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
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "hosted execution requires a Git checkout so code provenance can be proven"
        ) from exc


def _enforce_cli_budget_policy(target: float, emergency_threshold: float, reserve: float) -> None:
    if target > FROZEN_MAX_TARGET_RMB:
        raise ValueError(
            f"--target-rmb cannot exceed frozen normal-stop target {FROZEN_MAX_TARGET_RMB:.0f} RMB"
        )
    if emergency_threshold > FROZEN_MAX_EMERGENCY_THRESHOLD_RMB:
        raise ValueError(
            "--hard-cap-rmb cannot exceed frozen emergency stop threshold "
            f"{FROZEN_MAX_EMERGENCY_THRESHOLD_RMB:.0f} RMB"
        )
    if target <= 0 or emergency_threshold <= 0 or target >= emergency_threshold:
        raise ValueError("require 0 < target-rmb < hard-cap-rmb")
    if reserve <= 0 or reserve >= target:
        raise ValueError("request-reserve-rmb must be positive and below target-rmb")


def _enforce_first_launch_balance(
    *,
    available_rmb: float,
    minimum_initial_balance_rmb: float | None,
    ledger_preexisting: bool,
) -> None:
    if minimum_initial_balance_rmb is None or ledger_preexisting:
        return
    if minimum_initial_balance_rmb <= 0:
        raise ValueError("--minimum-initial-balance-rmb must be positive when supplied")
    if available_rmb < minimum_initial_balance_rmb:
        raise BudgetExceeded(
            "Refusing to start before any generation call: "
            f"available balance {available_rmb:.4f} RMB is below required "
            f"minimum {minimum_initial_balance_rmb:.2f} RMB."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a frozen FairEval hosted plan through Zhizengzeng with a "
            "200 RMB normal stop and 250 RMB emergency stop threshold. The gateway "
            "balance is reconciled around every persisted cell; this client-side "
            "threshold is not represented as an atomic provider-side spend cap. "
            "Dry-run by default."
        )
    )
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--freeze-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--ledger", default="results/budget/hosted_zzz_v1.json")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--family", action="append")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--target-rmb", type=float, default=200.0)
    parser.add_argument("--hard-cap-rmb", type=float, default=250.0)
    parser.add_argument("--request-reserve-rmb", type=float, default=2.0)
    parser.add_argument(
        "--minimum-initial-balance-rmb",
        type=float,
        help=(
            "Optional pre-spend gate. On a new ledger, refuse all generation calls "
            "unless the live available balance is at least this value."
        ),
    )
    parser.add_argument("--code-commit-sha")
    parser.add_argument(
        "--preexecution-seal",
        default="results/preexecution/seal-v1/PREEXECUTION_SEAL.json",
    )
    parser.add_argument(
        "--seal-plan-key",
        default="hosted_fairsynth",
        help="Plan block to verify inside the pre-execution seal.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    _enforce_cli_budget_policy(
        args.target_rmb,
        args.hard_cap_rmb,
        args.request_reserve_rmb,
    )
    _load_dotenv(Path(args.env_file))
    os.environ["FAIREVAL_HOSTED_GATEWAY"] = "zhizengzeng"
    os.environ.setdefault("ZZZ_BASE_URL", "https://api.zhizengzeng.com/v1")

    plan_dir = Path(args.plan_dir)
    output_jsonl = Path(args.output_jsonl)
    ledger_path = Path(args.ledger)
    families = set(args.family) if args.family else None
    cells, manifest = load_and_verify_plan(plan_dir)
    completed = completed_cell_ids(output_jsonl)
    ledger_preexisting = ledger_path.is_file()
    if completed and not ledger_preexisting:
        raise RuntimeError(
            "Refusing to resume hosted execution with completed output rows but no "
            "existing budget ledger. Restore the matching ledger instead of resetting "
            "spend accounting."
        )
    selected = pending_cells(cells, completed=completed, families=families)
    if args.max_cells is not None:
        if args.max_cells <= 0:
            raise ValueError("--max-cells must be positive")
        selected = selected[: args.max_cells]

    seal_path = Path(args.preexecution_seal)
    dry_report = {
        "schema_version": "faireval-hosted-budgeted-launch-v3",
        "mode": "execute" if args.execute else "dry_run",
        "gateway": "zhizengzeng",
        "plan_sha256": manifest.get("plan_sha256"),
        "planned_cells_total": len(cells),
        "completed_cells": len(completed),
        "selected_pending_cells": len(selected),
        "family_filter": None if families is None else sorted(families),
        "target_rmb": args.target_rmb,
        "emergency_stop_threshold_rmb": args.hard_cap_rmb,
        "request_reserve_rmb": args.request_reserve_rmb,
        "minimum_initial_balance_rmb": args.minimum_initial_balance_rmb,
        "ledger_preexisting": ledger_preexisting,
        "maximum_allowed_target_rmb": FROZEN_MAX_TARGET_RMB,
        "maximum_allowed_emergency_stop_threshold_rmb": FROZEN_MAX_EMERGENCY_THRESHOLD_RMB,
        "provider_side_atomic_spend_cap_claimed": False,
        "preexecution_seal_path": str(seal_path),
        "preexecution_seal_exists": seal_path.is_file(),
        "preexecution_seal_plan_key": args.seal_plan_key,
        "api_calls_made": 0,
    }
    if not args.execute:
        print(json.dumps(dry_report, indent=2, sort_keys=True))
        return 0

    if not args.code_commit_sha:
        raise ValueError("--code-commit-sha is required with --execute")
    checked_out_sha = _git_head()
    if args.code_commit_sha.strip() != checked_out_sha:
        raise ValueError(
            "--code-commit-sha must equal the currently checked-out Git HEAD; "
            f"argument={args.code_commit_sha.strip()} HEAD={checked_out_sha}. "
            "Do not mix code revisions inside a frozen hosted run log."
        )

    seal_verification = verify_preexecution_seal(
        seal_path,
        expected_commit_sha=checked_out_sha,
        plan_dir=plan_dir,
        plan_key=args.seal_plan_key,
    )

    api_key = os.environ.get("ZZZ_API_KEY")
    if not api_key:
        raise RuntimeError("ZZZ_API_KEY is required for hosted execution")

    # Pre-spend gate: paid execution cannot start unless the live account exposes
    # every exact frozen model ID. This model-list check makes no generation call.
    model_gate = verify_frozen_gateway_models(
        models_yaml=Path(args.models),
        api_key=api_key,
        base_url=os.environ.get("ZZZ_BASE_URL"),
    )
    if not model_gate["all_exact_models_available"]:
        raise RuntimeError(
            "Hosted execution blocked before spend: exact frozen gateway model IDs "
            f"are missing: {model_gate['missing_frozen_models']}"
        )

    ledger_path = Path(args.ledger)
    ledger_preexisting = ledger_path.is_file()
    guard = ZhizengzengBudgetGuard(
        ledger_path=ledger_path,
        target_rmb=args.target_rmb,
        hard_cap_rmb=args.hard_cap_rmb,
        request_reserve_rmb=args.request_reserve_rmb,
    )
    initial = guard.ensure_initialized()
    _enforce_first_launch_balance(
        available_rmb=initial.available_rmb,
        minimum_initial_balance_rmb=args.minimum_initial_balance_rmb,
        ledger_preexisting=ledger_preexisting,
    )
    if initial.spent_rmb >= args.target_rmb:
        raise BudgetExceeded(
            f"Refusing to start: existing experiment spend {initial.spent_rmb:.4f} RMB "
            f"already reaches normal stop target {args.target_rmb:.2f} RMB"
        )

    try:
        summary = execute_plan(
            plan_dir=plan_dir,
            freeze_root=Path(args.freeze_root),
            output_jsonl=output_jsonl,
            code_commit_sha=checked_out_sha,
            families=families,
            max_cells=args.max_cells,
            before_cell=guard.before_cell,
            after_cell=guard.after_cell,
        )
    except BudgetExceeded as exc:
        report = guard.report()
        report["execution_status"] = "STOPPED_BY_BUDGET_GUARD"
        report["reason"] = str(exc)
        report["model_gate"] = model_gate
        report["checked_out_code_commit_sha"] = checked_out_sha
        report["preexecution_seal"] = seal_verification
        report["provider_side_atomic_spend_cap_claimed"] = False
        print(json.dumps(report, indent=2, sort_keys=True))
        return 10

    report = guard.report()
    summary["budget"] = report
    summary["model_gate"] = model_gate
    summary["checked_out_code_commit_sha"] = checked_out_sha
    summary["preexecution_seal"] = seal_verification
    summary["provider_side_atomic_spend_cap_claimed"] = False
    summary["execution_status"] = "COMPLETED_SELECTED_CELLS"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
