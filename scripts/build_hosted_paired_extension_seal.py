from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from faireval.execute import load_and_verify_plan
from faireval.freeze import file_sha256
from scripts.build_preexecution_seal import (
    ROOT,
    _git,
    _json_digest,
    assert_clean_scientific_worktree,
    collect_spec_hashes,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a zero-call V4 pre-execution seal for the hosted FairSynth "
            "paired-completion extension."
        )
    )
    parser.add_argument(
        "--extension-plan-dir",
        default="results/plans/hosted-fairsynth-paired-extension-v1",
    )
    parser.add_argument(
        "--parent-plan-dir",
        default="results/plans/hosted-fairsynth-lean-v1",
    )
    parser.add_argument(
        "--output-dir",
        default="results/preexecution/hosted-paired-extension-seal-v1",
    )
    args = parser.parse_args()

    assert_clean_scientific_worktree()

    extension_dir = Path(args.extension_plan_dir)
    parent_dir = Path(args.parent_plan_dir)
    extension_rows, extension_manifest = load_and_verify_plan(extension_dir)
    parent_rows, parent_manifest = load_and_verify_plan(parent_dir)

    if extension_manifest.get("schema_version") != "faireval-hosted-paired-extension-plan-v1":
        raise ValueError("unexpected hosted extension plan schema")
    if extension_manifest.get("scientific_outcomes_inspected_before_extension_definition") is not False:
        raise ValueError("hosted extension plan is not outcome-blind")
    if extension_manifest.get("selection_scope") != "outcome_blind_coverage_and_cost_only":
        raise ValueError("hosted extension selection scope drift")
    if str(extension_manifest.get("parent_plan_sha256")) != str(parent_manifest.get("plan_sha256")):
        raise ValueError("hosted extension parent-plan hash mismatch")

    target_group_counts = extension_manifest.get("target_identity_group_counts")
    if target_group_counts != {"A": 3, "B": 3, "C": 3}:
        raise ValueError(
            "hosted paired extension seal expects exactly 3 users per synthetic identity group"
        )
    if int(extension_manifest.get("target_users_total", -1)) != 9:
        raise ValueError("hosted paired extension seal expects exactly 9 target users")
    if int(extension_manifest.get("target_cells_total_in_parent_plan", -1)) != 270:
        raise ValueError("hosted paired extension target geometry must be 270 total cells")
    if len(extension_rows) != int(extension_manifest.get("planned_api_cells", -1)):
        raise ValueError("hosted extension plan row count mismatch")

    if extension_manifest.get("run_schema_version") != "faireval-run-v6":
        raise ValueError("hosted extension must freeze faireval-run-v6")
    if extension_manifest.get("prompt_interface_version") != "faireval-prompt-interface-v6":
        raise ValueError("hosted extension must freeze faireval-prompt-interface-v6")
    for row in extension_rows:
        if row.get("run_schema_version") != "faireval-run-v6":
            raise ValueError("hosted extension row run schema drift")
        if row.get("prompt_interface_version") != "faireval-prompt-interface-v6":
            raise ValueError("hosted extension row prompt interface drift")

    spec_hashes = collect_spec_hashes()
    seal: dict[str, Any] = {
        "schema_version": "faireval-preexecution-seal-v4",
        "git_commit_sha": _git("rev-parse", "HEAD"),
        "git_branch_or_detached": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "scientific_worktree_clean": True,
        "scientific_spec_sha256": _json_digest(spec_hashes),
        "scientific_spec_file_count": len(spec_hashes),
        "scientific_spec_files": spec_hashes,
        "plans": {
            "hosted_extension": {
                "manifest_path": str(extension_dir / "plan_manifest.json"),
                "manifest_sha256": file_sha256(extension_dir / "plan_manifest.json"),
                "plan_sha256": extension_manifest.get("plan_sha256"),
                "planned_cells": len(extension_rows),
                "model_families": extension_manifest.get("model_families"),
            },
            "hosted_parent": {
                "manifest_path": str(parent_dir / "plan_manifest.json"),
                "manifest_sha256": file_sha256(parent_dir / "plan_manifest.json"),
                "plan_sha256": parent_manifest.get("plan_sha256"),
                "planned_cells": len(parent_rows),
                "model_families": parent_manifest.get("model_families"),
            },
        },
        "hosted_extension_manifest": {
            "path": str(extension_dir / "plan_manifest.json"),
            "sha256": file_sha256(extension_dir / "plan_manifest.json"),
            "target_users_total": extension_manifest.get("target_users_total"),
            "target_identity_group_counts": target_group_counts,
            "target_cells_total_in_parent_plan": extension_manifest.get(
                "target_cells_total_in_parent_plan"
            ),
            "target_cells_completed_before_extension": extension_manifest.get(
                "target_cells_completed_before_extension"
            ),
            "planned_extension_calls": len(extension_rows),
            "projected_incremental_cost_rmb": extension_manifest.get(
                "projected_incremental_cost_rmb"
            ),
            "projected_incremental_cost_with_safety_rmb": extension_manifest.get(
                "projected_incremental_cost_with_safety_rmb"
            ),
            "extension_budget_target_rmb": extension_manifest.get(
                "extension_budget_target_rmb"
            ),
            "extension_budget_hard_cap_rmb": extension_manifest.get(
                "extension_budget_hard_cap_rmb"
            ),
        },
        "prior_hosted_pilot_known": True,
        "prior_hosted_scientific_outcomes_inspected": False,
        "hosted_extension_selection_outcome_blind": True,
        "hosted_extension_results_seen_before_seal": False,
        "local_v7_results_known": True,
        "hosted_api_generation_calls_made": 0,
        "local_model_weights_loaded": False,
        "interpretation": (
            "The hosted paired-completion target was selected from prior plan coverage "
            "and observed cost only. No prior hosted rankings, utilities, effect sizes, "
            "or p-values were inspected before this seal."
        ),
    }
    seal["seal_sha256_without_self"] = _json_digest(seal)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seal_path = output_dir / "PREEXECUTION_SEAL.json"
    seal_path.write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "seal": str(seal_path),
        "git_commit_sha": seal["git_commit_sha"],
        "planned_extension_calls": len(extension_rows),
        "target_users_total": 9,
        "target_identity_group_counts": target_group_counts,
        "projected_incremental_cost_with_safety_rmb": seal[
            "hosted_extension_manifest"
        ]["projected_incremental_cost_with_safety_rmb"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
