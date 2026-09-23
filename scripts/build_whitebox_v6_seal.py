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
    from scripts.build_preexecution_seal import (
        assert_clean_scientific_worktree,
        collect_spec_hashes,
    )
except ModuleNotFoundError:
    from build_preexecution_seal import (
        assert_clean_scientific_worktree,
        collect_spec_hashes,
    )


ROOT = Path(__file__).resolve().parents[1]
V4_FROZEN_COMMIT = "05d90102a3de2c90310eee76b75757d745d6c664"
V5_FAILED_COMMIT = "b768a183a646a10909b240943751c9555f73d9da"
EXPECTED_CELLS = 144
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
        raise RuntimeError(
            f"V6 seal prerequisite failed: {label}; see {log_path}"
        )
    return {
        "label": label,
        "returncode": completed.returncode,
        "log": str(log_path.relative_to(ROOT)),
        "log_sha256": _sha256(log_path),
    }


def _assert_no_v6_results(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    populated = [
        path for path in run_dir.rglob("*")
        if path.is_file() and path.stat().st_size > 0
    ]
    if populated:
        raise RuntimeError(
            "cannot create a preregistered V6 canary seal after V6 run output exists: "
            + ", ".join(str(path) for path in populated[:10])
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the zero-generation pre-execution seal for FairEval V6 canary."
    )
    parser.add_argument(
        "--output-dir",
        default="results/preexecution/seal-v11",
    )
    parser.add_argument(
        "--plan-dir",
        default="results/plans/whitebox-v6-canary/core",
    )
    parser.add_argument(
        "--run-dir",
        default="results/runs/whitebox-v6-canary",
    )
    args = parser.parse_args()

    assert_clean_scientific_worktree()
    _assert_no_v6_results(ROOT / args.run_dir)

    output_dir = (ROOT / args.output_dir).resolve()
    command_dir = output_dir / "commands"
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        (
            "build_v6_canary_plan",
            [
                "scripts/build_whitebox_v6_canary.py",
                "--output-root",
                str((ROOT / "results/plans/whitebox-v6-canary").resolve()),
            ],
        ),
        (
            "v6_protocol_tests",
            [
                "-m",
                "pytest",
                "-q",
                "tests/test_output_protocol_v5.py",
                "tests/test_runner.py",
                "tests/test_run_audit_v5.py",
                "tests/test_prompt_and_validation.py",
                "tests/test_execute.py",
                "tests/test_preexecution_verification.py",
            ],
        ),
        (
            "whitebox_environment",
            ["scripts/check_whitebox_environment.py"],
        ),
    ]
    step_results = [_run(label, command, command_dir) for label, command in steps]
    assert_clean_scientific_worktree()
    _assert_no_v6_results(ROOT / args.run_dir)

    plan_dir = ROOT / args.plan_dir
    plan_rows, plan_manifest = load_and_verify_plan(plan_dir)
    if plan_manifest.get("schema_version") != "faireval-whitebox-v6-canary-plan-v1":
        raise RuntimeError("unexpected V6 canary plan schema")
    if len(plan_rows) != EXPECTED_CELLS:
        raise RuntimeError(f"V6 canary seal expected {EXPECTED_CELLS} cells")
    families = {str(row["model_family"]) for row in plan_rows}
    if families != EXPECTED_FAMILIES:
        raise RuntimeError(f"V6 canary family mismatch: {sorted(families)}")

    gate = plan_manifest.get("promotion_gate")
    if not isinstance(gate, dict):
        raise RuntimeError("V6 canary manifest lacks promotion gate")
    if float(gate.get("semantic_exact_k_candidate_valid_rate_per_model_min", -1)) != 0.95:
        raise RuntimeError("V6 canary promotion threshold must be frozen at 0.95")
    if gate.get("predeclared_before_canary_results") is not True:
        raise RuntimeError("V6 canary gate was not marked predeclared")

    environment_path = ROOT / "results/whitebox/environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if environment.get("status") != "PASS":
        raise RuntimeError("canonical V6 environment check did not pass")

    spec_hashes = collect_spec_hashes()
    seal: dict[str, Any] = {
        "schema_version": "faireval-preexecution-seal-v3",
        "seal_purpose": "v6_protocol_recovery_canary",
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
        "v4_frozen_commit": V4_FROZEN_COMMIT,
        "v5_failed_commit": V5_FAILED_COMMIT,
        "v4_artifacts_modified": False,
        "v6_canary_disjoint_from_v5_users": True,
        "v6_canary_results_seen_before_seal": False,
        "hosted_api_generation_calls_made": 0,
        "local_model_weights_loaded": False,
        "plans": {
            "whitebox_v6_canary": {
                "manifest_path": str((plan_dir / "plan_manifest.json").relative_to(ROOT)),
                "manifest_sha256": _sha256(plan_dir / "plan_manifest.json"),
                "plan_sha256": plan_manifest.get("plan_sha256"),
                "run_plan_file_sha256": plan_manifest.get("run_plan_file_sha256"),
                "planned_cells": len(plan_rows),
                "model_families": sorted(families),
            }
        },
        "promotion_gate": gate,
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
            "Run exactly the sealed 144-cell V6 canary. Do not scale if either model "
            "has semantic exact-k candidate validity below 0.95 or any hard gate fails."
        ),
    }
    seal["seal_sha256_without_self"] = _json_digest(seal)

    seal_path = output_dir / "PREEXECUTION_SEAL.json"
    seal_path.write_text(
        json.dumps(seal, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "seal": str(seal_path.relative_to(ROOT)),
                "seal_sha256": _sha256(seal_path),
                "git_commit_sha": seal["git_commit_sha"],
                "scientific_spec_sha256": seal["scientific_spec_sha256"],
                "planned_cells": EXPECTED_CELLS,
                "promotion_gate": gate,
                "v6_canary_results_seen_before_seal": False,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
