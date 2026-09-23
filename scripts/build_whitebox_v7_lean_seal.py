from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from faireval.execute import load_and_verify_plan
from faireval.freeze import file_sha256

try:
    from scripts.build_preexecution_seal import assert_clean_scientific_worktree, collect_spec_hashes
except ModuleNotFoundError:
    from build_preexecution_seal import assert_clean_scientific_worktree, collect_spec_hashes


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CELLS = 2880
EXPECTED_FAMILIES = {"qwen25_local", "phi35_local"}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run(label: str, command: list[str], command_dir: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, *command],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    command_dir.mkdir(parents=True, exist_ok=True)
    log_path = command_dir / f"{label}.log"
    log_path.write_text(
        "$ " + sys.executable + " " + " ".join(command) + "\n\n"
        + completed.stdout
        + ("\nSTDERR:\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"V7 lean seal prerequisite failed: {label}; see {log_path}")
    return {
        "label": label,
        "returncode": completed.returncode,
        "log": str(log_path.relative_to(ROOT)),
        "log_sha256": _sha256(log_path),
    }


def _assert_no_main_results(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    populated = [p for p in run_dir.rglob("*") if p.is_file() and p.stat().st_size > 0]
    if populated:
        raise RuntimeError(
            "cannot seal V7 lean main run after main output exists: "
            + ", ".join(str(p) for p in populated[:10])
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Seal the complete 2,880-cell FairEval V7 lean main run.")
    parser.add_argument("--output-dir", default="results/preexecution/seal-v13")
    parser.add_argument("--plan-dir", default="results/plans/whitebox-v7-lean/core")
    parser.add_argument("--run-dir", default="results/runs/whitebox-v7-lean")
    parser.add_argument(
        "--canary-report",
        default="results/analysis/whitebox-v7-canary/canary_gate.json",
    )
    args = parser.parse_args()

    assert_clean_scientific_worktree()
    _assert_no_main_results(ROOT / args.run_dir)

    canary_path = ROOT / args.canary_report
    if not canary_path.is_file():
        raise RuntimeError("V7 lean main seal requires completed V7 canary audit")
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    if canary.get("status") != "PASS" or canary.get("promotion_allowed") is not True:
        raise RuntimeError("V7 canary did not pass; main run remains blocked")
    families = canary.get("families", {})
    for family in EXPECTED_FAMILIES:
        report = families.get(family, {})
        if report.get("promotion_subgate_pass") is not True:
            raise RuntimeError(f"V7 canary family subgate failed: {family}")
        if float(report.get("semantic_valid_rate", 0.0)) < 0.95:
            raise RuntimeError(f"V7 canary semantic-validity threshold failed: {family}")

    output_dir = (ROOT / args.output_dir).resolve()
    command_dir = output_dir / "commands"
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        (
            "build_v7_lean_plan",
            [
                "scripts/build_whitebox_v7_lean_plan.py",
                "--output-root",
                str((ROOT / "results/plans/whitebox-v7-lean").resolve()),
            ],
        ),
        (
            "v7_protocol_tests",
            [
                "-m", "pytest", "-q",
                "tests/test_output_protocol_v5.py",
                "tests/test_runner.py",
                "tests/test_runner_v7.py",
                "tests/test_run_audit_v5.py",
                "tests/test_run_audit_v6.py",
                "tests/test_run_audit_v7.py",
                "tests/test_prompt_and_validation.py",
                "tests/test_execute.py",
                "tests/test_preexecution_verification.py",
            ],
        ),
        ("whitebox_environment", ["scripts/check_whitebox_environment.py"]),
    ]
    step_results = [_run(label, command, command_dir) for label, command in steps]
    assert_clean_scientific_worktree()
    _assert_no_main_results(ROOT / args.run_dir)

    plan_dir = ROOT / args.plan_dir
    rows, manifest = load_and_verify_plan(plan_dir)
    if manifest.get("schema_version") != "faireval-whitebox-v7-lean-plan-v1":
        raise RuntimeError("unexpected V7 lean plan schema")
    if len(rows) != EXPECTED_CELLS:
        raise RuntimeError(f"V7 lean plan must contain exactly {EXPECTED_CELLS} cells")
    plan_families = {str(row["model_family"]) for row in rows}
    if plan_families != EXPECTED_FAMILIES:
        raise RuntimeError(f"V7 lean family mismatch: {sorted(plan_families)}")
    if manifest.get("outcome_dependent_truncation") is not False:
        raise RuntimeError("V7 lean main run must be a complete, non-truncated plan")
    if manifest.get("protocol_canary_user_overlap") != []:
        raise RuntimeError("V7 lean main users overlap protocol canary users")

    environment_path = ROOT / "results/whitebox/environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if environment.get("status") != "PASS":
        raise RuntimeError("canonical local environment check did not pass")

    spec_hashes = collect_spec_hashes()
    seal: dict[str, Any] = {
        "schema_version": "faireval-preexecution-seal-v5",
        "seal_purpose": "v7_deadline_bounded_complete_2880_cell_main_run",
        "git_commit_sha": _git("rev-parse", "HEAD"),
        "git_branch_or_detached": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "scientific_worktree_clean": True,
        "scientific_spec_sha256": _json_digest(spec_hashes),
        "scientific_spec_file_count": len(spec_hashes),
        "scientific_spec_files": spec_hashes,
        "prior_v4_results_known": True,
        "prior_v4_protocol_failure_known": True,
        "prior_v5_canary_results_known": True,
        "prior_v5_canary_failed": True,
        "prior_v6_canary_results_known": True,
        "prior_v6_canary_failed": True,
        "v7_canary_results_known": True,
        "v7_canary_passed": True,
        "v7_main_users_disjoint_from_all_canaries": True,
        "v7_main_results_seen_before_seal": False,
        "hosted_api_generation_calls_made": 0,
        "local_model_weights_loaded": False,
        "canary_gate": {
            "path": str(canary_path.relative_to(ROOT)),
            "sha256": _sha256(canary_path),
            "status": canary.get("status"),
            "promotion_allowed": canary.get("promotion_allowed"),
            "families": canary.get("families"),
        },
        "plans": {
            "whitebox_v7_lean": {
                "manifest_path": str((plan_dir / "plan_manifest.json").relative_to(ROOT)),
                "manifest_sha256": _sha256(plan_dir / "plan_manifest.json"),
                "plan_sha256": manifest.get("plan_sha256"),
                "run_plan_file_sha256": manifest.get("run_plan_file_sha256"),
                "planned_cells": len(rows),
                "model_families": sorted(plan_families),
            }
        },
        "environment_check": {
            "path": str(environment_path.relative_to(ROOT)),
            "sha256": _sha256(environment_path),
            "status": environment.get("status"),
            "python": environment.get("python"),
            "torch": environment.get("torch"),
            "cuda_version": environment.get("cuda_version"),
            "gpu": environment.get("gpu"),
            "packages": environment.get("packages"),
        },
        "zero_generation_steps": step_results,
        "execution_rule": (
            "Execute all 2,880 sealed V7 cells. Primary local analysis requires complete "
            "coverage; no outcome-dependent early stopping or generative repair is allowed."
        ),
    }
    seal["seal_sha256_without_self"] = _json_digest(seal)
    seal_path = output_dir / "PREEXECUTION_SEAL.json"
    seal_path.write_text(
        json.dumps(seal, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "seal": str(seal_path.relative_to(ROOT)),
        "seal_sha256": _sha256(seal_path),
        "git_commit_sha": seal["git_commit_sha"],
        "planned_cells": EXPECTED_CELLS,
        "v7_canary_passed": True,
        "v7_main_results_seen_before_seal": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
