from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from faireval.execute import load_and_verify_plan
from faireval.freeze import canonical_json, file_sha256, load_frozen_instances


GROUPS = ("A", "B", "C")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def _completed_ids(path: Path) -> set[str]:
    return {
        str(row["planned_cell_id"])
        for row in _read_jsonl(path)
        if row.get("planned_cell_id")
    }


def _plan_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


def _load_operational_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("operational summary must be a JSON object")
    if payload.get("scientific_outcomes_inspected") is not False:
        raise ValueError(
            "paired extension must be defined before hosted scientific outcomes are inspected"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an outcome-blind hosted FairSynth paired-completion extension. "
            "Selection may use prior completion coverage/cost but never rankings or effects."
        )
    )
    parser.add_argument(
        "--parent-plan-dir",
        default="results/plans/hosted-fairsynth-lean-v1",
    )
    parser.add_argument(
        "--base-run",
        default="results/runs/hosted-fairsynth-lean-v1.jsonl",
    )
    parser.add_argument(
        "--freeze-root",
        default="data/frozen",
    )
    parser.add_argument(
        "--operational-summary",
        default="provenance/hosted_fairsynth_pilot99_operational_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plans/hosted-fairsynth-paired-extension-v1",
    )
    parser.add_argument("--users-per-identity-group", type=int, default=3)
    parser.add_argument("--budget-target-rmb", type=float, default=195.0)
    parser.add_argument("--budget-hard-cap-rmb", type=float, default=200.0)
    parser.add_argument("--cost-safety-multiplier", type=float, default=1.15)
    args = parser.parse_args()

    if args.users_per_identity_group <= 0:
        raise ValueError("--users-per-identity-group must be positive")
    if not (0 < args.budget_target_rmb < args.budget_hard_cap_rmb):
        raise ValueError("require 0 < budget-target-rmb < budget-hard-cap-rmb")
    if args.cost_safety_multiplier < 1.0:
        raise ValueError("--cost-safety-multiplier must be >= 1")

    parent_rows, parent_manifest = load_and_verify_plan(Path(args.parent_plan_dir))
    base_run = Path(args.base_run)
    if not base_run.is_file():
        raise FileNotFoundError(base_run)
    completed = _completed_ids(base_run)
    parent_ids = {str(row["cell_id"]) for row in parent_rows}
    unknown = completed - parent_ids
    if unknown:
        raise ValueError(f"base run contains IDs outside parent plan: {sorted(unknown)[:3]}")

    summary = _load_operational_summary(Path(args.operational_summary))
    if str(summary.get("plan_sha256")) != str(parent_manifest.get("plan_sha256")):
        raise ValueError("operational summary plan hash does not match parent hosted plan")

    synth = load_frozen_instances(Path(args.freeze_root) / "fairsynth360")
    group_by_user = {
        str(instance.user_id): str(instance.demographics.get("synthetic_identity_group"))
        for instance in synth
    }

    user_order: list[str] = []
    seen_users: set[str] = set()
    for row in parent_rows:
        user = str(row["user_id"])
        if user not in seen_users:
            seen_users.add(user)
            user_order.append(user)
    order_index = {user: i for i, user in enumerate(user_order)}

    completed_by_user: Counter[str] = Counter()
    for row in parent_rows:
        if str(row["cell_id"]) in completed:
            completed_by_user[str(row["user_id"])] += 1

    target_users: list[str] = []
    target_user_records: list[dict[str, Any]] = []
    per_group = int(args.users_per_identity_group)
    for group in GROUPS:
        candidates = [u for u in user_order if group_by_user.get(u) == group]
        # Scientific extension users must be completely untouched by the
        # historical hosted run. This keeps every inferential row under one
        # execution commit/specification; the 99 historical rows remain
        # operational provenance only.
        untouched = [u for u in candidates if completed_by_user[u] == 0]
        untouched.sort(key=lambda u: order_index[u])
        chosen = untouched[:per_group]
        if len(chosen) != per_group:
            raise ValueError(f"identity group {group} has insufficient hosted-plan users")
        target_users.extend(chosen)
        for user in chosen:
            target_user_records.append(
                {
                    "user_id": user,
                    "synthetic_identity_group": group,
                    "completed_cells_before_extension": int(completed_by_user[user]),
                    "historically_untouched_user": bool(completed_by_user[user] == 0),
                    "parent_plan_order": int(order_index[user]),
                }
            )

    target_user_set = set(target_users)
    analysis_condition_rows = [
        row
        for row in parent_rows
        if str(row["user_id"]) in target_user_set
        and (
            str(row["condition"]["condition_id"]) in {"C1", "C3", "C4"}
            or str(row["condition"]["condition_id"]).startswith("C2:")
        )
    ]
    target_rows = analysis_condition_rows
    pending_rows = [
        row for row in target_rows if str(row["cell_id"]) not in completed
    ]

    expected_target_cells = len(target_users) * 5 * 6
    if len(target_rows) != expected_target_cells:
        raise RuntimeError(
            f"target geometry drift: expected {expected_target_cells}, got {len(target_rows)}"
        )

    target_group_counts = Counter(group_by_user[user] for user in target_users)
    if target_group_counts != Counter({g: per_group for g in GROUPS}):
        raise RuntimeError(f"target identity balance failed: {target_group_counts}")

    # Estimate only from outcome-blind observed request-window costs.
    fam_summary = summary.get("families")
    if not isinstance(fam_summary, dict):
        raise ValueError("operational summary lacks per-family cost information")
    mean_cost: dict[str, float] = {}
    all_cost = 0.0
    all_n = 0
    for family, raw in fam_summary.items():
        if not isinstance(raw, dict):
            continue
        n = int(raw.get("cost_observations", 0))
        cost = float(raw.get("observed_request_window_cost_rmb", 0.0))
        if n > 0:
            mean_cost[str(family)] = cost / n
            all_cost += cost
            all_n += n
    if all_n <= 0:
        raise ValueError("no observed request-window cost observations available")
    fallback_mean = all_cost / all_n

    projected = 0.0
    pending_by_family: Counter[str] = Counter()
    for row in pending_rows:
        family = str(row["model_family"])
        pending_by_family[family] += 1
        projected += mean_cost.get(family, fallback_mean)
    projected_with_safety = projected * float(args.cost_safety_multiplier)
    if projected_with_safety > float(args.budget_target_rmb):
        raise RuntimeError(
            "paired target does not fit the outcome-blind extension budget estimate: "
            f"projected={projected:.2f} RMB, with_safety={projected_with_safety:.2f} RMB, "
            f"target={args.budget_target_rmb:.2f} RMB"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "run_plan.jsonl"
    plan_path.write_text(
        "".join(canonical_json(row) + "\n" for row in pending_rows),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema_version": "faireval-hosted-paired-extension-plan-v1",
        "selection_scope": "outcome_blind_coverage_and_cost_only",
        "parent_plan_dir": str(Path(args.parent_plan_dir)),
        "parent_plan_sha256": parent_manifest.get("plan_sha256"),
        "base_run_path": str(base_run),
        "base_run_sha256": file_sha256(base_run),
        "operational_summary_path": str(Path(args.operational_summary)),
        "operational_summary_sha256": file_sha256(Path(args.operational_summary)),
        "scientific_outcomes_inspected_before_extension_definition": False,
        "selection_rule": (
            "within each synthetic identity group, select only users with zero historical "
            "hosted cells, in immutable parent-plan order; no historical scientific rows "
            "are reused in the paired extension"
        ),
        "included_condition_ids": ["C1", "C2:*", "C3", "C4"],
        "preference_only_c0_included": False,
        "users_per_identity_group": per_group,
        "target_users": target_user_records,
        "target_identity_group_counts": dict(sorted(target_group_counts.items())),
        "target_users_total": len(target_users),
        "target_cells_total_in_parent_plan": len(target_rows),
        "target_cells_completed_before_extension": len(target_rows) - len(pending_rows),
        "historically_touched_users_excluded": sorted(
            user for user in user_order if completed_by_user[user] > 0
        ),
        "planned_api_cells": len(pending_rows),
        "pending_cells_by_family": dict(sorted(pending_by_family.items())),
        "observed_mean_request_window_cost_rmb_by_family": {
            family: round(value, 6) for family, value in sorted(mean_cost.items())
        },
        "projected_incremental_cost_rmb": round(projected, 6),
        "cost_safety_multiplier": float(args.cost_safety_multiplier),
        "projected_incremental_cost_with_safety_rmb": round(projected_with_safety, 6),
        "extension_budget_target_rmb": float(args.budget_target_rmb),
        "extension_budget_hard_cap_rmb": float(args.budget_hard_cap_rmb),
        "model_families": sorted({str(row["model_family"]) for row in target_rows}),
        "plan_sha256": _plan_digest(pending_rows),
    }
    manifest["run_plan_file_sha256"] = file_sha256(plan_path)
    (output_dir / "plan_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "target_users.json").write_text(
        json.dumps(target_user_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
