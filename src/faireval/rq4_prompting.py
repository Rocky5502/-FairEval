from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .analysis import _pair_payload, _pairing_key
from .freeze import canonical_json


RQ4_DATASETS = {"movielens_1m", "lastfm_1k"}


def _hash_row(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(row)).encode("utf-8")).hexdigest()


def build_identity_irrelevance_plan(
    source_cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive an immutable RQ4 prompting plan without mutating the audit plan.

    Only observed C1 and primary confirmatory gender C2 cells are copied. The
    semantic recommendation task, candidate order, model, decoding settings, and
    repetition stay frozen; only ``prompt_mode`` changes from ``audit`` to
    ``identity_irrelevance``. The original cell ID is retained for provenance.
    """
    output: list[dict[str, Any]] = []
    for cell in source_cells:
        if str(cell.get("dataset")) not in RQ4_DATASETS:
            continue
        roles = {str(value) for value in cell.get("analysis_roles", [])}
        if "rq4_mitigation_baseline" not in roles:
            continue
        if str(cell.get("prompt_mode")) != "audit":
            raise ValueError("RQ4 prompting source cells must come from the neutral audit plan")
        condition = cell.get("condition")
        if not isinstance(condition, Mapping):
            raise ValueError("run-plan cell has malformed condition")
        condition_id = str(condition.get("condition_id", ""))
        if condition_id == "C1":
            pass
        elif condition_id.startswith("C2:"):
            intervention = condition.get("intervention", {})
            if not isinstance(intervention, Mapping):
                raise ValueError("C2 intervention metadata is malformed")
            if intervention.get("attribute") != "gender":
                continue
            if cell.get("confirmatory") is not True:
                continue
        else:
            continue

        row = dict(cell)
        source_cell_id = str(row.pop("cell_id"))
        row["schema_version"] = "faireval-rq4-prompt-plan-v1"
        row["source_audit_cell_id"] = source_cell_id
        row["prompt_mode"] = "identity_irrelevance"
        row["confirmatory"] = False
        row["analysis_roles"] = sorted(roles | {"rq4_identity_irrelevance_prompting"})
        row["rq4_intervention"] = "identity_irrelevance_prompting"
        row["cell_id"] = _hash_row(row)
        output.append(row)

    if not output:
        raise ValueError("source plan contains no eligible RQ4 C1/C2 cells")
    ids = [str(row["cell_id"]) for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("derived RQ4 prompting plan contains duplicate cell IDs")
    return sorted(
        output,
        key=lambda row: (
            str(row["dataset"]),
            str(row["model_family"]),
            str(row["user_id"]),
            str(row["condition"]["condition_id"]),
            int(row["repetition"]),
        ),
    )


def build_identity_irrelevance_pairs(
    aggregated_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build user-level observed-vs-counterfactual pairs for the RQ4 prompt baseline."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in aggregated_rows:
        if str(row.get("dataset")) not in RQ4_DATASETS:
            continue
        if str(row.get("prompt_mode")) != "identity_irrelevance":
            continue
        roles = {str(value) for value in row.get("analysis_roles", [])}
        if "rq4_identity_irrelevance_prompting" not in roles:
            continue
        grouped[_pairing_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for rows in grouped.values():
        observed = [row for row in rows if str(row.get("condition_id")) == "C1"]
        counterfactual = [
            row for row in rows
            if str(row.get("condition_id", "")).startswith("C2:")
            and isinstance(row.get("condition_intervention"), Mapping)
            and row["condition_intervention"].get("attribute") == "gender"
        ]
        if not observed and not counterfactual:
            continue
        if len(observed) != 1 or len(counterfactual) != 1:
            raise ValueError(
                "RQ4 identity-irrelevance analysis requires exactly one C1 and one primary gender C2 per user/model"
            )
        left, right = observed[0], counterfactual[0]
        output.append(
            _pair_payload(
                left,
                right,
                rq="RQ4-PROMPT",
                contrast="identity_irrelevance_observed_vs_counterfactual",
                extra={
                    "attribute": "gender",
                    "mitigation": "identity_irrelevance_prompting",
                    "confirmatory_fairness_family": False,
                },
            )
        )
    if not output:
        raise ValueError("no RQ4 identity-irrelevance pairs found")
    return output


def summarize_prompting_pairs(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Produce descriptive dataset/model strata plus an equal-stratum macro summary."""
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[
            (
                str(row["dataset"]),
                str(row["model_family"]),
                str(row["requested_model_id"]),
            )
        ].append(row)

    strata: list[dict[str, Any]] = []
    for (dataset, family, model_id), rows in sorted(grouped.items()):
        utilities = [
            (float(row["left_ndcg"]) + float(row["right_ndcg"])) / 2.0
            for row in rows
        ]
        abs_cug = [abs(float(row["left_ndcg"]) - float(row["right_ndcg"])) for row in rows]
        invalid = [
            (float(row["left_invalid_rate"]) + float(row["right_invalid_rate"])) / 2.0
            for row in rows
        ]
        strata.append(
            {
                "dataset": dataset,
                "model_family": family,
                "requested_model_id": model_id,
                "n_users": len(rows),
                "mean_identity_ndcg": statistics.fmean(utilities),
                "mean_abs_cug_ndcg": statistics.fmean(abs_cug),
                "mean_invalid_rate": statistics.fmean(invalid),
            }
        )

    return {
        "schema_version": "faireval-rq4-prompting-summary-v1",
        "intervention": "identity_irrelevance_prompting",
        "unit": "user_then_equal_dataset_model_stratum",
        "strata": strata,
        "macro": {
            "n_strata": len(strata),
            "mean_identity_ndcg": statistics.fmean(row["mean_identity_ndcg"] for row in strata),
            "mean_abs_cug_ndcg": statistics.fmean(row["mean_abs_cug_ndcg"] for row in strata),
            "mean_invalid_rate": statistics.fmean(row["mean_invalid_rate"] for row in strata),
        },
        "confirmatory_p_values": False,
    }
