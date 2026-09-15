from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .execute import load_and_verify_plan
from .freeze import load_frozen_instances
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k
from .run_audit import audit_run_log


UTILITY_METRICS = ("ndcg", "recall", "mrr")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def score_run_log(
    output_jsonl: Path,
    *,
    plan_dir: Path,
    freeze_root: Path,
    invalid_utility_policy: str = "zero",
) -> list[dict[str, Any]]:
    """Convert audited model outputs into deterministic run-level utility rows.

    Primary end-to-end utility uses ``invalid_utility_policy='zero'``: a
    persistent invalid response delivers no usable recommendation and therefore
    receives zero nDCG/Recall/MRR. We simultaneously retain ``valid_only_*``
    values as a sensitivity view and an explicit invalid indicator, preventing
    complete-case-only reporting from hiding model failure.
    """
    if invalid_utility_policy != "zero":
        raise ValueError("the frozen ECIR primary policy currently supports only invalid_utility_policy='zero'")

    audit_run_log(output_jsonl, plan_dir=plan_dir)
    plan_rows, manifest = load_and_verify_plan(plan_dir)
    plan_by_id = {str(row["cell_id"]): row for row in plan_rows}
    run_rows = _read_jsonl(output_jsonl)

    datasets = sorted({str(row["dataset"]) for row in run_rows})
    instance_index: dict[tuple[str, str], Any] = {}
    for dataset in datasets:
        for instance in load_frozen_instances(freeze_root / dataset):
            key = (dataset, str(instance.user_id))
            if key in instance_index:
                raise ValueError(f"duplicate frozen instance {key!r}")
            instance_index[key] = instance

    scored: list[dict[str, Any]] = []
    for row in run_rows:
        cell_id = str(row["planned_cell_id"])
        planned = plan_by_id[cell_id]
        dataset = str(row["dataset"])
        user_id = str(row["user_id"])
        instance = instance_index.get((dataset, user_id))
        if instance is None:
            raise ValueError(f"missing frozen instance {(dataset, user_id)!r}")

        k = int(planned["k"])
        final_valid = bool(row["final_valid"])
        ranking = [str(value) for value in (row.get("ranking") or [])]
        relevant = instance.relevant_item_ids

        if final_valid:
            utility = {
                "ndcg": ndcg_at_k(ranking, relevant, k),
                "recall": recall_at_k(ranking, relevant, k),
                "mrr": mrr_at_k(ranking, relevant, k),
            }
            valid_only = dict(utility)
        else:
            utility = {metric: 0.0 for metric in UTILITY_METRICS}
            valid_only = {metric: None for metric in UTILITY_METRICS}

        condition = planned["condition"]
        if not isinstance(condition, Mapping):
            raise ValueError(f"cell {cell_id}: planned condition is malformed")
        intervention = condition.get("intervention")
        if not isinstance(intervention, Mapping):
            intervention = {}

        scored.append(
            {
                "schema_version": "faireval-scored-run-v1",
                "plan_sha256": manifest.get("plan_sha256"),
                "planned_cell_id": cell_id,
                "dataset": dataset,
                "user_id": user_id,
                "model_family": str(row["model_family"]),
                "requested_model_id": str(row["requested_model_id"]),
                "condition_id": str(row["condition_id"]),
                "condition_name": str(row["condition_name"]),
                "condition_intervention": dict(intervention),
                "analysis_roles": list(planned.get("analysis_roles", [])),
                "confirmatory": bool(planned.get("confirmatory", False)),
                "prompt_template_id": str(row["prompt_template_id"]),
                "prompt_mode": str(row["prompt_mode"]),
                "cue_id": str(row["cue_id"]),
                "candidate_order_seed": row.get("candidate_order_seed"),
                "k": k,
                "repetition": int(row["repetition"]),
                "final_valid": final_valid,
                "invalid_output": not final_valid,
                "ndcg": float(utility["ndcg"]),
                "recall": float(utility["recall"]),
                "mrr": float(utility["mrr"]),
                "valid_only_ndcg": valid_only["ndcg"],
                "valid_only_recall": valid_only["recall"],
                "valid_only_mrr": valid_only["mrr"],
                "code_commit_sha": str(row["code_commit_sha"]),
            }
        )
    return scored


def aggregate_repetitions(scored_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Average repeated generations within a frozen user-condition-model cell."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        intervention_json = json.dumps(
            row.get("condition_intervention", {}),
            sort_keys=True,
            separators=(",", ":"),
        )
        key = (
            row["dataset"],
            row["user_id"],
            row["model_family"],
            row["requested_model_id"],
            row["condition_id"],
            row["condition_name"],
            intervention_json,
            bool(row["confirmatory"]),
            tuple(row.get("analysis_roles", [])),
            row["prompt_template_id"],
            row["prompt_mode"],
            row["cue_id"],
            row.get("candidate_order_seed"),
            int(row["k"]),
            row["code_commit_sha"],
            row.get("plan_sha256"),
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        (
            dataset,
            user_id,
            family,
            model_id,
            condition_id,
            condition_name,
            intervention_json,
            confirmatory,
            analysis_roles,
            template_id,
            prompt_mode,
            cue_id,
            candidate_order_seed,
            k,
            code_commit_sha,
            plan_sha256,
        ) = key
        result: dict[str, Any] = {
            "schema_version": "faireval-user-condition-v1",
            "dataset": dataset,
            "user_id": user_id,
            "model_family": family,
            "requested_model_id": model_id,
            "condition_id": condition_id,
            "condition_name": condition_name,
            "condition_intervention": json.loads(intervention_json),
            "confirmatory": confirmatory,
            "analysis_roles": list(analysis_roles),
            "prompt_template_id": template_id,
            "prompt_mode": prompt_mode,
            "cue_id": cue_id,
            "candidate_order_seed": candidate_order_seed,
            "k": k,
            "repetitions": len(rows),
            "invalid_rate": sum(bool(row["invalid_output"]) for row in rows) / len(rows),
            "code_commit_sha": code_commit_sha,
            "plan_sha256": plan_sha256,
        }
        for metric in UTILITY_METRICS:
            result[metric] = sum(float(row[metric]) for row in rows) / len(rows)
            valid_values = [
                float(row[f"valid_only_{metric}"])
                for row in rows
                if row[f"valid_only_{metric}"] is not None
            ]
            result[f"valid_only_{metric}"] = (
                None if not valid_values else sum(valid_values) / len(valid_values)
            )
        output.append(result)

    return sorted(
        output,
        key=lambda row: (
            str(row["dataset"]),
            str(row["model_family"]),
            str(row["user_id"]),
            str(row["condition_id"]),
        ),
    )


def _pairing_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["dataset"],
        row["user_id"],
        row["model_family"],
        row["requested_model_id"],
        row["prompt_template_id"],
        row["prompt_mode"],
        row["cue_id"],
        row.get("candidate_order_seed"),
        row["k"],
        row["code_commit_sha"],
        row.get("plan_sha256"),
    )


def _pair_payload(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    rq: str,
    contrast: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "faireval-paired-estimand-v1",
        "rq": rq,
        "contrast": contrast,
        "dataset": left["dataset"],
        "user_id": left["user_id"],
        "model_family": left["model_family"],
        "requested_model_id": left["requested_model_id"],
        "prompt_template_id": left["prompt_template_id"],
        "prompt_mode": left["prompt_mode"],
        "cue_id": left["cue_id"],
        "candidate_order_seed": left.get("candidate_order_seed"),
        "k": left["k"],
        "left_condition_id": left["condition_id"],
        "right_condition_id": right["condition_id"],
        "left_invalid_rate": left["invalid_rate"],
        "right_invalid_rate": right["invalid_rate"],
        "invalid_rate_difference": left["invalid_rate"] - right["invalid_rate"],
        "code_commit_sha": left["code_commit_sha"],
        "plan_sha256": left.get("plan_sha256"),
    }
    for metric in UTILITY_METRICS:
        payload[f"left_{metric}"] = left[metric]
        payload[f"right_{metric}"] = right[metric]
        payload[f"delta_{metric}"] = left[metric] - right[metric]
        payload[f"left_valid_only_{metric}"] = left[f"valid_only_{metric}"]
        payload[f"right_valid_only_{metric}"] = right[f"valid_only_{metric}"]
    if extra:
        payload.update(extra)
    return payload


def build_rq1_confirmatory_pairs(
    aggregated_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair C1 observed-demographic utility with every confirmatory C2 alternative."""
    by_key: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in aggregated_rows:
        by_key[_pairing_key(row)].append(row)

    pairs: list[dict[str, Any]] = []
    for rows in by_key.values():
        observed = [row for row in rows if row["condition_id"] == "C1"]
        if len(observed) > 1:
            raise ValueError("multiple C1 rows found for one pairing key")
        if not observed:
            continue
        left = observed[0]
        for right in rows:
            if not str(right["condition_id"]).startswith("C2:"):
                continue
            if not bool(right["confirmatory"]):
                continue
            intervention = right.get("condition_intervention", {})
            if not isinstance(intervention, Mapping):
                raise ValueError("C2 intervention metadata is malformed")
            pairs.append(
                _pair_payload(
                    left,
                    right,
                    rq="RQ1",
                    contrast="observed_vs_demographic_counterfactual",
                    extra={
                        "attribute": intervention.get("attribute"),
                        "observed_value": intervention.get("observed_value"),
                        "counterfactual_value": intervention.get("counterfactual_value"),
                    },
                )
            )
    return pairs


def build_rq2_pairs(
    aggregated_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build true-vs-shuffled PVA and true-vs-preference-only personality contrasts."""
    by_key: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in aggregated_rows:
        condition_id = str(row["condition_id"])
        if condition_id not in {"C0", "C3", "C4"}:
            continue
        key = _pairing_key(row)
        if condition_id in by_key[key]:
            raise ValueError(f"duplicate {condition_id} row for one RQ2 pairing key")
        by_key[key][condition_id] = row

    pairs: list[dict[str, Any]] = []
    for conditions in by_key.values():
        true = conditions.get("C3")
        if true is None:
            continue
        shuffled = conditions.get("C4")
        if shuffled is not None:
            pairs.append(
                _pair_payload(
                    true,
                    shuffled,
                    rq="RQ2",
                    contrast="true_vs_shuffled_personality",
                )
            )
        preference = conditions.get("C0")
        if preference is not None:
            pairs.append(
                _pair_payload(
                    true,
                    preference,
                    rq="RQ2",
                    contrast="true_personality_vs_preference_only",
                )
            )
    return pairs
