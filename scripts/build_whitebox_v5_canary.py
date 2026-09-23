from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from faireval.freeze import canonical_json, file_sha256, verify_freeze
from faireval.local_plan import compile_local_open_weight_plan


V4_FROZEN_COMMIT = "05d90102a3de2c90310eee76b75757d745d6c664"
EXPECTED_USERS = 12
EXPECTED_CONDITIONS_PER_USER = 6
EXPECTED_REPETITIONS = 1
EXPECTED_FAMILIES = {"qwen25_local", "phi35_local"}
EXPECTED_CELLS_PER_FAMILY = 72
EXPECTED_TOTAL_CELLS = 144
PROMOTION_THRESHOLD = 0.95


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the immutable 144-cell FairEval V5 protocol-recovery canary plan."
    )
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument("--counterfactuals", default="configs/counterfactuals.yaml")
    parser.add_argument("--models", default="configs/local_models.yaml")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--output-root",
        default="results/plans/whitebox-v5-canary",
    )
    args = parser.parse_args()

    freeze_root = Path(args.freeze_root)
    freeze_verification = verify_freeze(freeze_root / "fairsynth360")
    if freeze_verification.get("verification") != "PASS":
        raise RuntimeError("FairSynth-360 freeze verification failed before V5 planning")
    if int(freeze_verification.get("verified_instance_count", -1)) != 360:
        raise RuntimeError(
            "V5 canary requires the canonical 360-user FairSynth freeze; "
            f"found {freeze_verification.get('verified_instance_count')}"
        )

    cells, base_manifest = compile_local_open_weight_plan(
        freeze_root=freeze_root,
        counterfactuals_yaml=Path(args.counterfactuals),
        local_models_yaml=Path(args.models),
        seed=args.seed,
        include_real_world=False,
        fairsynth_users=EXPECTED_USERS,
        repetitions=EXPECTED_REPETITIONS,
    )
    if len(cells) != EXPECTED_TOTAL_CELLS:
        raise AssertionError(
            f"V5 canary geometry drift: expected {EXPECTED_TOTAL_CELLS}, got {len(cells)}"
        )
    ids = [str(row["cell_id"]) for row in cells]
    if len(set(ids)) != len(ids):
        raise AssertionError("V5 canary plan contains duplicate cell IDs")

    families = Counter(str(row["model_family"]) for row in cells)
    if set(families) != EXPECTED_FAMILIES:
        raise AssertionError(f"V5 canary family drift: {sorted(families)}")
    if any(families[family] != EXPECTED_CELLS_PER_FAMILY for family in EXPECTED_FAMILIES):
        raise AssertionError(f"V5 canary family cell-count drift: {dict(families)}")
    if any(str(row.get("dataset")) != "fairsynth360" for row in cells):
        raise AssertionError("V5 canary must contain FairSynth-360 only")
    if any(int(row.get("k", -1)) != 10 for row in cells):
        raise AssertionError("V5 canary requires k=10")
    if any(int(row.get("repetition", -1)) != 0 for row in cells):
        raise AssertionError("V5 canary requires exactly one repetition")

    users = {str(row["user_id"]) for row in cells}
    if len(users) != EXPECTED_USERS:
        raise AssertionError(f"V5 canary expected {EXPECTED_USERS} users, got {len(users)}")

    per_user_family: Counter[tuple[str, str]] = Counter(
        (str(row["user_id"]), str(row["model_family"])) for row in cells
    )
    if any(value != EXPECTED_CONDITIONS_PER_USER for value in per_user_family.values()):
        raise AssertionError("V5 canary expected six condition cells per user/model")

    output_dir = Path(args.output_root) / "core"
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "run_plan.jsonl"
    plan_path.write_text(
        "".join(canonical_json(row) + "\n" for row in cells),
        encoding="utf-8",
        newline="\n",
    )
    plan_sha = hashlib.sha256(canonical_json(cells).encode("utf-8")).hexdigest()

    promotion_gate = {
        "predeclared_before_canary_results": True,
        "semantic_exact_k_candidate_valid_rate_per_model_min": PROMOTION_THRESHOLD,
        "required_cells_per_model": EXPECTED_CELLS_PER_FAMILY,
        "zero_duplicate_planned_cell_ids": True,
        "zero_candidate_id_mutation": True,
        "zero_parser_ambiguity": True,
        "preexecution_seal_pass": True,
        "commit_provenance_pass": True,
        "hardware_model_revision_checks_pass": True,
        "failure_action": "do_not_scale_diagnose_failed_72_cell_model",
    }
    manifest = {
        **base_manifest,
        "schema_version": "faireval-whitebox-v5-canary-plan-v1",
        "protocol_recovery_after_frozen_v4_failure": True,
        "v4_frozen_commit": V4_FROZEN_COMMIT,
        "canary_results_seen_when_gate_defined": False,
        "fairsynth_freeze_verification": {
            "verification": freeze_verification.get("verification"),
            "verified_instance_count": freeze_verification.get("verified_instance_count"),
            "verified_instances_sha256": freeze_verification.get("verified_instances_sha256"),
        },
        "fairsynth_users": EXPECTED_USERS,
        "conditions_per_user": EXPECTED_CONDITIONS_PER_USER,
        "repetitions": EXPECTED_REPETITIONS,
        "cells_per_family": EXPECTED_CELLS_PER_FAMILY,
        "planned_api_cells": EXPECTED_TOTAL_CELLS,
        "plan_sha256": plan_sha,
        "run_plan_file_sha256": file_sha256(plan_path),
        "promotion_gate": promotion_gate,
        "execution_policy": {
            "one_model_family_per_process": True,
            "one_generation_per_cell": True,
            "generative_format_repair": False,
            "deterministic_envelope_normalization_only": True,
            "never_write_v4_output_directory": True,
        },
    }
    manifest_path = output_dir / "plan_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "READY",
                "plan_dir": str(output_dir),
                "planned_cells": EXPECTED_TOTAL_CELLS,
                "cells_per_family": dict(sorted(families.items())),
                "plan_sha256": plan_sha,
                "run_plan_file_sha256": manifest["run_plan_file_sha256"],
                "promotion_gate": promotion_gate,
                "model_weights_loaded": False,
                "hosted_api_calls_made": 0,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
