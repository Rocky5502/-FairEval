from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from faireval.freeze import canonical_json, file_sha256, verify_freeze
from faireval.local_plan import compile_local_open_weight_plan


EXPECTED_USERS = 120
USER_OFFSET = 36
EXPECTED_CONDITIONS_PER_USER = 6
EXPECTED_REPETITIONS = 2
EXPECTED_FAMILIES = {"qwen25_local", "phi35_local"}
EXPECTED_CELLS_PER_FAMILY = 1440
EXPECTED_TOTAL_CELLS = 2880
CANARY_USER_COUNT = 36


def _version_cells(cells):
    output = []
    for source in cells:
        row = dict(source)
        row.pop("cell_id", None)
        row["run_schema_version"] = "faireval-run-v7"
        row["prompt_interface_version"] = "faireval-prompt-interface-v7"
        row["cell_id"] = hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the complete 2,880-cell deadline-bounded FairEval V7 local plan."
    )
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument("--counterfactuals", default="configs/counterfactuals.yaml")
    parser.add_argument("--models", default="configs/local_models.yaml")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output-root", default="results/plans/whitebox-v7-lean")
    args = parser.parse_args()

    freeze_root = Path(args.freeze_root)
    verification = verify_freeze(freeze_root / "fairsynth360")
    if verification.get("verification") != "PASS":
        raise RuntimeError("FairSynth-360 freeze verification failed")
    if int(verification.get("verified_instance_count", -1)) != 360:
        raise RuntimeError("V7 lean plan requires the canonical 360-user FairSynth freeze")

    cells, base_manifest = compile_local_open_weight_plan(
        freeze_root=freeze_root,
        counterfactuals_yaml=Path(args.counterfactuals),
        local_models_yaml=Path(args.models),
        seed=args.seed,
        include_real_world=False,
        fairsynth_users=EXPECTED_USERS,
        fairsynth_user_offset=USER_OFFSET,
        repetitions=EXPECTED_REPETITIONS,
    )
    cells = _version_cells(cells)

    if len(cells) != EXPECTED_TOTAL_CELLS:
        raise AssertionError(
            f"V7 lean geometry drift: expected {EXPECTED_TOTAL_CELLS}, got {len(cells)}"
        )
    ids = [str(row["cell_id"]) for row in cells]
    if len(ids) != len(set(ids)):
        raise AssertionError("V7 lean plan contains duplicate cell IDs")

    users = {str(row["user_id"]) for row in cells}
    if len(users) != EXPECTED_USERS:
        raise AssertionError(f"expected {EXPECTED_USERS} users, got {len(users)}")

    families = Counter(str(row["model_family"]) for row in cells)
    if set(families) != EXPECTED_FAMILIES:
        raise AssertionError(f"family drift: {dict(families)}")
    if any(families[f] != EXPECTED_CELLS_PER_FAMILY for f in EXPECTED_FAMILIES):
        raise AssertionError(f"per-family geometry drift: {dict(families)}")

    per_user_family = Counter(
        (str(row["user_id"]), str(row["model_family"])) for row in cells
    )
    expected_per_user_family = EXPECTED_CONDITIONS_PER_USER * EXPECTED_REPETITIONS
    if any(v != expected_per_user_family for v in per_user_family.values()):
        raise AssertionError("each user/model must contain six conditions x two repetitions")

    prior_cells, _ = compile_local_open_weight_plan(
        freeze_root=freeze_root,
        counterfactuals_yaml=Path(args.counterfactuals),
        local_models_yaml=Path(args.models),
        seed=args.seed,
        include_real_world=False,
        fairsynth_users=CANARY_USER_COUNT,
        fairsynth_user_offset=0,
        repetitions=1,
    )
    prior_users = {str(row["user_id"]) for row in prior_cells}
    overlap = sorted(users & prior_users)
    if overlap:
        raise AssertionError(f"main V7 users overlap protocol canaries: {overlap}")

    output_dir = Path(args.output_root) / "core"
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "run_plan.jsonl"
    plan_path.write_text(
        "".join(canonical_json(row) + "\n" for row in cells),
        encoding="utf-8",
        newline="\n",
    )
    plan_sha = hashlib.sha256(canonical_json(cells).encode("utf-8")).hexdigest()
    manifest = {
        **base_manifest,
        "schema_version": "faireval-whitebox-v7-lean-plan-v1",
        "deadline_bounded_complete_matrix": True,
        "outcome_dependent_truncation": False,
        "fairsynth_users": EXPECTED_USERS,
        "fairsynth_user_offset": USER_OFFSET,
        "protocol_canary_users_excluded": CANARY_USER_COUNT,
        "protocol_canary_user_overlap": overlap,
        "conditions_per_user": EXPECTED_CONDITIONS_PER_USER,
        "repetitions": EXPECTED_REPETITIONS,
        "cells_per_family": EXPECTED_CELLS_PER_FAMILY,
        "planned_api_cells": EXPECTED_TOTAL_CELLS,
        "model_families": sorted(EXPECTED_FAMILIES),
        "run_schema_version": "faireval-run-v7",
        "prompt_interface_version": "faireval-prompt-interface-v7",
        "plan_sha256": plan_sha,
        "run_plan_file_sha256": file_sha256(plan_path),
        "fairsynth_freeze_verification": {
            "verification": verification.get("verification"),
            "verified_instance_count": verification.get("verified_instance_count"),
            "verified_instances_sha256": verification.get("verified_instances_sha256"),
        },
        "execution_policy": {
            "one_model_family_per_process": True,
            "one_generation_per_cell": True,
            "generative_format_repair": False,
            "deterministic_envelope_normalization_only": True,
            "complete_plan_required_for_primary_local_analysis": True,
        },
    }
    manifest_path = output_dir / "plan_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "READY",
        "plan_dir": str(output_dir),
        "users": EXPECTED_USERS,
        "repetitions": EXPECTED_REPETITIONS,
        "planned_cells": EXPECTED_TOTAL_CELLS,
        "cells_per_family": dict(sorted(families.items())),
        "protocol_canary_user_overlap": overlap,
        "plan_sha256": plan_sha,
        "run_plan_file_sha256": manifest["run_plan_file_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
