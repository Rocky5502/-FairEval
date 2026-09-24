from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from faireval.datasets.fairsynth360 import FairSynth360Adapter
from faireval.freeze import canonical_json, file_sha256, freeze_dataset
from faireval.hosted_plan import compile_hosted_fairsynth_plan
from scripts.build_hosted_paired_extension import main


ROOT = Path(__file__).resolve().parents[1]


def test_hosted_paired_extension_reuses_only_full_users_and_excludes_partial(
    tmp_path: Path, monkeypatch
) -> None:
    freeze_root = tmp_path / "frozen"
    freeze_dataset(
        FairSynth360Adapter(),
        tmp_path / "unused",
        freeze_root / "fairsynth360",
        users=90,
        candidate_set_size=30,
        max_history_items=8,
        seed=1729,
    )
    cells, manifest = compile_hosted_fairsynth_plan(
        freeze_root=freeze_root,
        models_yaml=ROOT / "configs" / "models.yaml",
        users=30,
        repetitions=1,
        seed=1729,
    )

    parent = tmp_path / "parent"
    parent.mkdir()
    plan_path = parent / "run_plan.jsonl"
    plan_path.write_text(
        "".join(canonical_json(row) + "\n" for row in cells),
        encoding="utf-8",
    )
    manifest["plan_sha256"] = hashlib.sha256(
        canonical_json(cells).encode("utf-8")
    ).hexdigest()
    manifest["run_plan_file_sha256"] = file_sha256(plan_path)
    (parent / "plan_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Parent plan is 36 cells/user. Persist two complete users + 27 cells of a
    # third, mirroring the real 99-cell operational prefix. The scientific
    # extension must exclude all three touched users.
    user_order = []
    seen = set()
    for row in cells:
        user = str(row["user_id"])
        if user not in seen:
            seen.add(user)
            user_order.append(user)
    complete_users = set(user_order[:2])
    partial_user = user_order[2]

    completed_rows = [
        row for row in cells if str(row["user_id"]) in complete_users
    ]
    completed_rows += [
        row for row in cells if str(row["user_id"]) == partial_user
    ][:27]
    base_run = tmp_path / "base.jsonl"
    base_run.write_text(
        "".join(
            json.dumps({"planned_cell_id": row["cell_id"]}) + "\n"
            for row in completed_rows
        ),
        encoding="utf-8",
    )

    per_family = {}
    for family in ("openai", "anthropic", "google", "deepseek", "qwen", "meta"):
        per_family[family] = {
            "cells": 16,
            "users_touched": 3,
            "identity_complete_users": 3,
            "personality_complete_users": 2,
            "all_six_conditions_complete_users": 2,
            "observed_request_window_cost_rmb": 1.6,
            "cost_observations": 16,
        }
    operational = tmp_path / "operational.json"
    operational.write_text(
        json.dumps(
            {
                "schema_version": "faireval-hosted-pilot-operational-summary-v1",
                "plan_sha256": manifest["plan_sha256"],
                "scientific_outcomes_inspected": False,
                "families": per_family,
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "extension"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hosted_paired_extension.py",
            "--parent-plan-dir", str(parent),
            "--base-run", str(base_run),
            "--freeze-root", str(freeze_root),
            "--operational-summary", str(operational),
            "--output-dir", str(output),
            "--users-per-identity-group", "3",
            "--budget-target-rmb", "190",
            "--budget-hard-cap-rmb", "200",
        ],
    )
    assert main() == 0

    ext = json.loads((output / "plan_manifest.json").read_text(encoding="utf-8"))
    assert ext["target_users_total"] == 9
    assert ext["target_identity_group_counts"] == {"A": 3, "B": 3, "C": 3}
    assert partial_user in ext["partially_completed_users_excluded"]
    selected = {row["user_id"]: row for row in ext["target_users"]}
    assert partial_user not in selected
    assert complete_users.isdisjoint(selected)
    assert all(bool(row["historically_untouched_user"]) for row in ext["target_users"])
    assert set(ext["historically_touched_users_excluded"]) >= complete_users | {partial_user}
    # Nine untouched users x five inferential conditions x six hosted families.
    assert ext["planned_api_cells"] == 9 * 5 * 6
    assert ext["target_cells_total_in_parent_plan"] == 270
    assert ext["target_cells_completed_before_extension"] == 0
    assert ext["preference_only_c0_included"] is False
    assert ext["scientific_outcomes_inspected_before_extension_definition"] is False
