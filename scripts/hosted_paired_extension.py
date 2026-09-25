from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "results" / "plans" / "hosted-fairsynth-paired-extension-v1"
SEAL = ROOT / "results" / "preexecution" / "hosted-paired-extension-seal-v1" / "PREEXECUTION_SEAL.json"
BASE_RUN = ROOT / "results" / "runs" / "hosted-fairsynth-lean-v1.jsonl"
EXTENSION_RUN = ROOT / "results" / "runs" / "hosted-fairsynth-paired-extension-v1.jsonl"
LEDGER = ROOT / "results" / "budget" / "hosted_fairsynth_paired_extension_v1.json"
FINAL_ZIP = ROOT / "dist" / "FairEval_ECIR2027_Overleaf_HOSTED_PAIRED.zip"

TARGET_RMB = 195.0
EMERGENCY_RMB = 200.0
REQUEST_RESERVE_RMB = 2.0
MINIMUM_FIRST_LAUNCH_BALANCE_RMB = 200.0


def _run(args: list[str], *, allow_budget_stop: bool = False) -> int:
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode == 10 and allow_budget_stop:
        return 10
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with code {result.returncode}: "
            + " ".join([sys.executable, *args])
        )
    return 0


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _load_manifest() -> dict:
    path = PLAN_DIR / "plan_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"extension manifest missing: {path}. Run --prepare first."
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("extension plan manifest must be a JSON object")
    return value


def _assert_geometry(manifest: dict) -> None:
    expected = {
        "target_users_total": 9,
        "target_cells_total_in_parent_plan": 270,
        "target_cells_completed_before_extension": 0,
        "planned_api_cells": 270,
    }
    for field, wanted in expected.items():
        actual = int(manifest.get(field, -1))
        if actual != wanted:
            raise RuntimeError(
                f"hosted extension geometry drift: {field} expected {wanted}, got {actual}"
            )
    if manifest.get("target_identity_group_counts") != {"A": 3, "B": 3, "C": 3}:
        raise RuntimeError("hosted extension identity balance must be exactly A/B/C = 3/3/3")
    if manifest.get("scientific_outcomes_inspected_before_extension_definition") is not False:
        raise RuntimeError("hosted extension no longer satisfies outcome-blind selection")
    projected = float(manifest.get("projected_incremental_cost_with_safety_rmb", 1e9))
    if projected > TARGET_RMB:
        raise RuntimeError(
            f"hosted extension safety-adjusted projection {projected:.2f} RMB exceeds {TARGET_RMB:.0f} RMB"
        )


def prepare() -> int:
    if not BASE_RUN.is_file():
        raise FileNotFoundError(
            f"historical 99-cell run not found: {BASE_RUN}. Use the repo copy containing "
            "the original hosted pilot artifacts."
        )

    _run(["scripts/build_hosted_paired_extension.py"])
    manifest = _load_manifest()
    _assert_geometry(manifest)
    _run(["scripts/build_hosted_paired_extension_seal.py"])

    # Dry-run checks plan/CLI wiring only. It does not query models, spend balance,
    # or make generation calls.
    _run([
        "scripts/run_hosted_budgeted.py",
        "--plan-dir", str(PLAN_DIR.relative_to(ROOT)),
        "--freeze-root", "data/frozen",
        "--output-jsonl", str(EXTENSION_RUN.relative_to(ROOT)),
        "--ledger", str(LEDGER.relative_to(ROOT)),
        "--target-rmb", f"{TARGET_RMB:g}",
        "--hard-cap-rmb", f"{EMERGENCY_RMB:g}",
        "--request-reserve-rmb", f"{REQUEST_RESERVE_RMB:g}",
        "--minimum-initial-balance-rmb", f"{MINIMUM_FIRST_LAUNCH_BALANCE_RMB:g}",
        "--preexecution-seal", str(SEAL.relative_to(ROOT)),
        "--seal-plan-key", "hosted_extension",
    ])

    print(json.dumps({
        "status": "PREPARED_ZERO_CALL",
        "git_commit_sha": _git_head(),
        "target_users": manifest["target_users_total"],
        "target_identity_group_counts": manifest["target_identity_group_counts"],
        "historical_scientific_cells_reused": manifest["target_cells_completed_before_extension"],
        "new_calls": manifest["planned_api_cells"],
        "projected_incremental_cost_rmb": manifest["projected_incremental_cost_rmb"],
        "projected_incremental_cost_with_safety_rmb": manifest[
            "projected_incremental_cost_with_safety_rmb"
        ],
        "normal_stop_target_rmb": TARGET_RMB,
        "emergency_threshold_rmb": EMERGENCY_RMB,
        "minimum_live_balance_for_first_launch_rmb": MINIMUM_FIRST_LAUNCH_BALANCE_RMB,
        "next": (
            "Fund the hosted account so live available balance is >=200 RMB, then run "
            "this script with --execute. Do not inspect hosted rankings/effects before execution."
        ),
    }, indent=2, sort_keys=True))
    return 0


def execute() -> int:
    manifest = _load_manifest()
    _assert_geometry(manifest)
    if not SEAL.is_file():
        raise FileNotFoundError(f"pre-execution seal missing: {SEAL}. Run --prepare first.")

    code = _run(
        [
            "scripts/run_hosted_budgeted.py",
            "--plan-dir", str(PLAN_DIR.relative_to(ROOT)),
            "--freeze-root", "data/frozen",
            "--output-jsonl", str(EXTENSION_RUN.relative_to(ROOT)),
            "--ledger", str(LEDGER.relative_to(ROOT)),
            "--target-rmb", f"{TARGET_RMB:g}",
            "--hard-cap-rmb", f"{EMERGENCY_RMB:g}",
            "--request-reserve-rmb", f"{REQUEST_RESERVE_RMB:g}",
            "--minimum-initial-balance-rmb", f"{MINIMUM_FIRST_LAUNCH_BALANCE_RMB:g}",
            "--preexecution-seal", str(SEAL.relative_to(ROOT)),
            "--seal-plan-key", "hosted_extension",
            "--code-commit-sha", _git_head(),
            "--execute",
        ],
        allow_budget_stop=True,
    )
    if code == 10:
        print(
            "Hosted extension stopped by the frozen budget guard. Existing rows and "
            "ledger are resumable. Do not delete or regenerate them."
        )
        return 10

    _run(["scripts/finalize_hosted_paired_extension.py"])
    print(json.dumps({
        "status": "HOSTED_PAIRED_EXTENSION_COMPLETE",
        "extension_run": str(EXTENSION_RUN.relative_to(ROOT)),
        "ledger": str(LEDGER.relative_to(ROOT)),
        "hosted_table": "paper/generated/fairsynth_hosted_table.tex",
        "hosted_summary": "paper/generated/fairsynth_hosted_summary.tex",
        "hosted_figure": "paper/figures/fairsynth_hosted_profiles.pdf",
        "overleaf_zip": str(FINAL_ZIP.relative_to(ROOT)),
    }, indent=2, sort_keys=True))
    return 0


def status() -> int:
    manifest = _load_manifest()
    _assert_geometry(manifest)
    completed = 0
    if EXTENSION_RUN.is_file():
        with EXTENSION_RUN.open("r", encoding="utf-8") as handle:
            completed = sum(1 for line in handle if line.strip())
    print(json.dumps({
        "status": "HOSTED_PAIRED_EXTENSION_STATUS",
        "git_commit_sha": _git_head(),
        "planned_new_calls": manifest["planned_api_cells"],
        "completed_new_rows": completed,
        "remaining_new_rows": int(manifest["planned_api_cells"]) - completed,
        "ledger_exists": LEDGER.is_file(),
        "seal_exists": SEAL.is_file(),
        "final_table_exists": (ROOT / "paper/generated/fairsynth_hosted_table.tex").is_file(),
        "final_figure_exists": (ROOT / "paper/figures/fairsynth_hosted_profiles.pdf").is_file(),
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, execute/resume, and finalize the sealed hosted FairSynth paired extension."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Zero-call plan + seal + dry-run.")
    mode.add_argument("--execute", action="store_true", help="Execute/resume paid calls, then finalize.")
    mode.add_argument("--status", action="store_true", help="Report extension progress without model calls.")
    args = parser.parse_args()

    if args.prepare:
        return prepare()
    if args.execute:
        return execute()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
