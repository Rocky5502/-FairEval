from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "results" / "plans" / "hosted-fairsynth-paired-balanced450-v1"
SEAL = ROOT / "results" / "preexecution" / "hosted-paired-balanced450-seal-v1" / "PREEXECUTION_SEAL.json"
BASE_RUN = ROOT / "results" / "runs" / "hosted-fairsynth-lean-v1.jsonl"
EXTENSION_RUN = ROOT / "results" / "runs" / "hosted-fairsynth-paired-balanced450-v1.jsonl"
LEDGER = ROOT / "results" / "budget" / "hosted_fairsynth_paired_balanced450_v1.json"
ANALYSIS_DIR = ROOT / "results" / "analysis" / "fairsynth-hosted-paired-balanced450-v1"
FINAL_ZIP = ROOT / "dist" / "FairEval_ECIR2027_Overleaf_HOSTED_BALANCED450.zip"

FAMILIES = ("meta", "qwen", "deepseek", "openai", "anthropic")
USERS_PER_IDENTITY_GROUP = 6
TARGET_USERS = USERS_PER_IDENTITY_GROUP * 3
TARGET_RMB = 170.0
EMERGENCY_RMB = 185.0
REQUEST_RESERVE_RMB = 2.0
MINIMUM_FIRST_LAUNCH_BALANCE_RMB = 185.0
EXPECTED_CALLS = TARGET_USERS * 5 * len(FAMILIES)


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
            f"balanced450 manifest missing: {path}. Run --prepare first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("balanced450 plan manifest must be a JSON object")
    return payload


def _assert_geometry(manifest: dict) -> None:
    expected = {
        "users_per_identity_group": USERS_PER_IDENTITY_GROUP,
        "target_users_total": TARGET_USERS,
        "target_cells_total_in_parent_plan": EXPECTED_CALLS,
        "target_cells_completed_before_extension": 0,
        "planned_api_cells": EXPECTED_CALLS,
    }
    for field, wanted in expected.items():
        actual = int(manifest.get(field, -1))
        if actual != wanted:
            raise RuntimeError(
                f"balanced450 geometry drift: {field} expected {wanted}, got {actual}"
            )

    expected_groups = {
        "A": USERS_PER_IDENTITY_GROUP,
        "B": USERS_PER_IDENTITY_GROUP,
        "C": USERS_PER_IDENTITY_GROUP,
    }
    if manifest.get("target_identity_group_counts") != expected_groups:
        raise RuntimeError(
            "balanced450 identity balance drift: "
            f"expected {expected_groups}, got {manifest.get('target_identity_group_counts')}"
        )

    if manifest.get("scientific_outcomes_inspected_before_extension_definition") is not False:
        raise RuntimeError("balanced450 selection is no longer outcome-blind")
    if manifest.get("run_schema_version") != "faireval-run-v7":
        raise RuntimeError("balanced450 run schema must remain faireval-run-v7")
    if manifest.get("prompt_interface_version") != "faireval-prompt-interface-v7":
        raise RuntimeError("balanced450 prompt interface must remain faireval-prompt-interface-v7")
    if set(manifest.get("model_families", [])) != set(FAMILIES):
        raise RuntimeError(
            f"balanced450 families drift: expected {sorted(FAMILIES)}, "
            f"got {manifest.get('model_families')}"
        )

    projected = float(manifest.get("projected_incremental_cost_with_safety_rmb", 1e9))
    if projected > TARGET_RMB:
        raise RuntimeError(
            f"balanced450 safety-adjusted projection {projected:.2f} RMB "
            f"exceeds frozen {TARGET_RMB:.0f} RMB target"
        )


def _runner_args(*, family: str | None = None, execute: bool = False) -> list[str]:
    args = [
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
    ]
    if family is not None:
        args.extend(["--family", family])
    if execute:
        args.extend(["--code-commit-sha", _git_head(), "--execute"])
    return args


def prepare() -> int:
    if not BASE_RUN.is_file():
        raise FileNotFoundError(
            f"historical 99-cell run not found: {BASE_RUN}. Restore the preserved pilot first."
        )

    planner = [
        "scripts/build_hosted_paired_extension.py",
        "--output-dir", str(PLAN_DIR.relative_to(ROOT)),
        "--users-per-identity-group", str(USERS_PER_IDENTITY_GROUP),
        "--budget-target-rmb", f"{TARGET_RMB:g}",
        "--budget-hard-cap-rmb", f"{EMERGENCY_RMB:g}",
        "--cost-safety-multiplier", "1.15",
    ]
    for family in FAMILIES:
        planner.extend(["--family", family])
    _run(planner)

    manifest = _load_manifest()
    _assert_geometry(manifest)

    _run([
        "scripts/build_hosted_paired_extension_seal.py",
        "--extension-plan-dir", str(PLAN_DIR.relative_to(ROOT)),
        "--output-dir", str(SEAL.parent.relative_to(ROOT)),
    ])

    _run(_runner_args())

    print(json.dumps({
        "status": "BALANCED450_PREPARED_ZERO_CALL",
        "git_commit_sha": _git_head(),
        "target_users": manifest["target_users_total"],
        "target_identity_group_counts": manifest["target_identity_group_counts"],
        "historical_scientific_cells_reused": manifest[
            "target_cells_completed_before_extension"
        ],
        "families": list(FAMILIES),
        "new_calls": manifest["planned_api_cells"],
        "projected_incremental_cost_rmb": manifest["projected_incremental_cost_rmb"],
        "projected_incremental_cost_with_safety_rmb": manifest[
            "projected_incremental_cost_with_safety_rmb"
        ],
        "normal_stop_target_rmb": TARGET_RMB,
        "emergency_threshold_rmb": EMERGENCY_RMB,
        "api_calls_made": 0,
    }, indent=2, sort_keys=True))
    return 0


def execute() -> int:
    manifest = _load_manifest()
    _assert_geometry(manifest)
    if not SEAL.is_file():
        raise FileNotFoundError(f"balanced450 pre-execution seal missing: {SEAL}")

    # Execute cheapest-to-most-expensive. Family order is cost-only and frozen
    # before any scientific hosted outcome is observed.
    for family in FAMILIES:
        code = _run(_runner_args(family=family, execute=True), allow_budget_stop=True)
        if code == 10:
            print(
                f"Balanced450 stopped safely by the frozen budget guard during {family}. "
                "Preserve JSONL + ledger and rerun --execute to resume."
            )
            return 10

    _run([
        "scripts/finalize_hosted_paired_extension.py",
        "--extension-plan-dir", str(PLAN_DIR.relative_to(ROOT)),
        "--extension-run", str(EXTENSION_RUN.relative_to(ROOT)),
        "--output-dir", str(ANALYSIS_DIR.relative_to(ROOT)),
        "--overleaf-zip", str(FINAL_ZIP.relative_to(ROOT)),
    ])

    expected_artifacts = [
        ROOT / "paper/generated/fairsynth_hosted_table.tex",
        ROOT / "paper/generated/fairsynth_hosted_profiles_table.tex",
        ROOT / "paper/generated/fairsynth_hosted_summary.tex",
        ROOT / "paper/figures/fairsynth_hosted_profiles.pdf",
        FINAL_ZIP,
    ]
    missing = [str(path) for path in expected_artifacts if not path.is_file()]
    if missing:
        raise RuntimeError(f"balanced450 finalization missing artifacts: {missing}")

    print(json.dumps({
        "status": "HOSTED_BALANCED450_COMPLETE",
        "families": list(FAMILIES),
        "target_users": TARGET_USERS,
        "target_cells": EXPECTED_CALLS,
        "extension_run": str(EXTENSION_RUN.relative_to(ROOT)),
        "ledger": str(LEDGER.relative_to(ROOT)),
        "analysis_manifest": str((ANALYSIS_DIR / "manifest.json").relative_to(ROOT)),
        "hosted_effect_table": "paper/generated/fairsynth_hosted_table.tex",
        "hosted_profiles_table": "paper/generated/fairsynth_hosted_profiles_table.tex",
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
        "status": "HOSTED_BALANCED450_STATUS",
        "git_commit_sha": _git_head(),
        "families": list(FAMILIES),
        "target_users": TARGET_USERS,
        "planned_new_calls": manifest["planned_api_cells"],
        "completed_new_rows": completed,
        "remaining_new_rows": int(manifest["planned_api_cells"]) - completed,
        "ledger_exists": LEDGER.is_file(),
        "seal_exists": SEAL.is_file(),
        "final_effect_table_exists": (
            ROOT / "paper/generated/fairsynth_hosted_table.tex"
        ).is_file(),
        "final_profiles_table_exists": (
            ROOT / "paper/generated/fairsynth_hosted_profiles_table.tex"
        ).is_file(),
        "final_figure_exists": (
            ROOT / "paper/figures/fairsynth_hosted_profiles.pdf"
        ).is_file(),
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Outcome-blind balanced hosted extension: 18 untouched FairSynth users "
            "(6/6/6 identity balance), five inferential conditions, five families, "
            "450 calls, with family selection based only on pre-outcome cost."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        return prepare()
    if args.execute:
        return execute()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
