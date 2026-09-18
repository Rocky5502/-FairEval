from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from faireval.execute import load_and_verify_plan


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit hosted execution coverage and observed balance movement without "
            "reading recommendation rankings or scientific outcome values."
        )
    )
    parser.add_argument("--plan-dir", default="results/plans/hosted-fairsynth-lean-v1")
    parser.add_argument("--output-jsonl", default="results/runs/hosted-fairsynth-lean-v1.jsonl")
    parser.add_argument("--ledger", default="results/budget/hosted_zzz_lean_v1.json")
    args = parser.parse_args()

    plan, manifest = load_and_verify_plan(Path(args.plan_dir))
    by_id = {str(row["cell_id"]): row for row in plan}
    run_rows = _read_jsonl(Path(args.output_jsonl))
    completed_ids = [str(row.get("planned_cell_id", "")) for row in run_rows if row.get("planned_cell_id")]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("run log contains duplicate planned_cell_id values")
    unknown = sorted(set(completed_ids) - set(by_id))
    if unknown:
        raise ValueError(f"run log contains cell IDs outside the plan: {unknown[:3]}")

    completed_plan = [by_id[cell_id] for cell_id in completed_ids]
    families = sorted({str(row["model_family"]) for row in plan})

    by_family_cells: dict[str, int] = defaultdict(int)
    by_family_users: dict[str, set[str]] = defaultdict(set)
    by_family_user_conditions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in completed_plan:
        family = str(row["model_family"])
        user = str(row["user_id"])
        condition = str(row["condition"]["condition_id"])
        by_family_cells[family] += 1
        by_family_users[family].add(user)
        by_family_user_conditions[(family, user)].add(condition)

    identity_complete: dict[str, int] = {}
    personality_complete: dict[str, int] = {}
    fully_complete_users: dict[str, int] = {}
    for family in families:
        users = by_family_users.get(family, set())
        n_id = 0
        n_p = 0
        n_full = 0
        for user in users:
            conditions = by_family_user_conditions[(family, user)]
            has_c1 = "C1" in conditions
            c2_count = sum(1 for value in conditions if value.startswith("C2:"))
            if has_c1 and c2_count >= 2:
                n_id += 1
            if "C3" in conditions and "C4" in conditions:
                n_p += 1
            if len(conditions) >= 6:
                n_full += 1
        identity_complete[family] = n_id
        personality_complete[family] = n_p
        fully_complete_users[family] = n_full

    ledger_path = Path(args.ledger)
    ledger = _load_json(ledger_path) if ledger_path.is_file() else {}
    checks = ledger.get("checks", []) if isinstance(ledger.get("checks"), list) else []
    before: dict[str, float] = {}
    observed_cost_by_cell: dict[str, float] = {}
    for row in checks:
        if not isinstance(row, dict):
            continue
        cell_id = row.get("cell_id")
        if not cell_id:
            continue
        event = row.get("event")
        available = row.get("available_rmb")
        if available is None:
            continue
        cell_id = str(cell_id)
        if event == "before_cell":
            before[cell_id] = float(available)
        elif event == "after_cell" and cell_id in before:
            observed_cost_by_cell[cell_id] = max(0.0, before[cell_id] - float(available))

    cost_by_family: dict[str, float] = defaultdict(float)
    cost_cells_by_family: dict[str, int] = defaultdict(int)
    for cell_id, cost in observed_cost_by_cell.items():
        planned = by_id.get(cell_id)
        if planned is None:
            continue
        family = str(planned["model_family"])
        cost_by_family[family] += cost
        cost_cells_by_family[family] += 1

    latest = ledger.get("latest") if isinstance(ledger.get("latest"), dict) else {}
    payload = {
        "schema_version": "faireval-hosted-progress-audit-v1",
        "plan_sha256": manifest.get("plan_sha256"),
        "planned_cells": len(plan),
        "completed_cells": len(completed_ids),
        "remaining_cells": len(plan) - len(completed_ids),
        "completion_fraction": 0.0 if not plan else len(completed_ids) / len(plan),
        "unique_completed_users": len({str(row["user_id"]) for row in completed_plan}),
        "cells_by_family": {family: by_family_cells.get(family, 0) for family in families},
        "users_touched_by_family": {family: len(by_family_users.get(family, set())) for family in families},
        "identity_complete_users_by_family": identity_complete,
        "personality_complete_users_by_family": personality_complete,
        "all_six_conditions_complete_users_by_family": fully_complete_users,
        "observed_request_window_cost_rmb_by_family": {
            family: round(cost_by_family.get(family, 0.0), 6) for family in families
        },
        "cost_observations_by_family": {
            family: cost_cells_by_family.get(family, 0) for family in families
        },
        "ledger_initial_balance_rmb": ledger.get("initial_balance_rmb"),
        "ledger_latest_available_rmb": latest.get("available_rmb"),
        "ledger_latest_spent_rmb": latest.get("spent_rmb"),
        "scientific_outcomes_inspected": False,
        "note": (
            "This audit summarizes plan coverage and gateway balance movement only. "
            "It does not inspect rankings, utilities, effect sizes, p-values, or fairness outcomes."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
