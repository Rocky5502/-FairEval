from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .analysis import UTILITY_METRICS, _matched_behavioral_diagnostics, _pairing_key


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _base_payload(left: Mapping[str, Any], *, rq: str, contrast: str) -> dict[str, Any]:
    return {
        "schema_version": "faireval-fairsynth-estimand-v1",
        "rq": rq,
        "contrast": contrast,
        "dataset": "fairsynth360",
        "user_id": left["user_id"],
        "model_family": left["model_family"],
        "requested_model_id": left["requested_model_id"],
        "prompt_template_id": left["prompt_template_id"],
        "prompt_mode": left["prompt_mode"],
        "cue_id": left["cue_id"],
        "candidate_order_seed": left.get("candidate_order_seed"),
        "k": int(left["k"]),
        "code_commit_sha": left["code_commit_sha"],
        "plan_sha256": left.get("plan_sha256"),
        "synthetic_control": True,
        "real_world_claim_allowed": False,
    }


def build_fairsynth_identity_pairs(
    aggregated_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reduce two A/B/C alternatives to one user-level identity sanity estimand.

    For each user/model cell the observed synthetic identity C1 is compared with
    the mean utility across all C2 alternative labels. This avoids pretending the
    two counterfactual labels from one user are independent observations.
    """
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in aggregated_rows:
        if row.get("dataset") == "fairsynth360":
            grouped[_pairing_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for rows in grouped.values():
        observed = [row for row in rows if row.get("condition_id") == "C1"]
        counterfactuals = [
            row for row in rows if str(row.get("condition_id", "")).startswith("C2:")
        ]
        if not observed or not counterfactuals:
            continue
        if len(observed) != 1:
            raise ValueError("FairSynth requires one C1 row per user/model pairing key")
        left = observed[0]
        payload = _base_payload(
            left,
            rq="SYNTH-ID",
            contrast="observed_identity_vs_mean_counterfactual_identity",
        )
        payload["left_condition_id"] = left["condition_id"]
        payload["right_condition_id"] = "mean_of_all_C2_identity_alternatives"
        payload["counterfactual_alternatives"] = len(counterfactuals)
        payload["left_invalid_rate"] = float(left["invalid_rate"])
        payload["right_invalid_rate"] = _mean(
            [float(row["invalid_rate"]) for row in counterfactuals]
        )
        payload["invalid_rate_difference"] = (
            payload["left_invalid_rate"] - payload["right_invalid_rate"]
        )
        for metric in UTILITY_METRICS:
            left_value = float(left[metric])
            right_value = _mean([float(row[metric]) for row in counterfactuals])
            payload[f"left_{metric}"] = left_value
            payload[f"right_{metric}"] = right_value
            payload[f"delta_{metric}"] = left_value - right_value

        # Behavioral diagnostics are descriptive. Average the observed-vs-each-C2
        # matched-repetition diagnostic, ignoring pairs with no jointly valid run.
        diagnostics = [
            _matched_behavioral_diagnostics(left, right) for right in counterfactuals
        ]
        for field in (
            "mean_rbo_at_k",
            "mean_jaccard_at_k",
            "mean_one_minus_rbo",
            "mean_one_minus_jaccard",
        ):
            values = [float(row[field]) for row in diagnostics if row[field] is not None]
            payload[field] = None if not values else _mean(values)
        payload["behavioral_valid_pair_count"] = sum(
            int(row["behavioral_valid_pair_count"]) for row in diagnostics
        )
        output.append(payload)
    return output


def build_fairsynth_personality_pairs(
    aggregated_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build true-synthetic-OCEAN versus shuffled-profile sanity pairs."""
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in aggregated_rows:
        if row.get("dataset") != "fairsynth360":
            continue
        condition_id = str(row.get("condition_id"))
        if condition_id not in {"C3", "C4"}:
            continue
        key = _pairing_key(row)
        if condition_id in grouped[key]:
            raise ValueError(f"duplicate FairSynth {condition_id} row for one pairing key")
        grouped[key][condition_id] = row

    output: list[dict[str, Any]] = []
    for conditions in grouped.values():
        left = conditions.get("C3")
        right = conditions.get("C4")
        if left is None or right is None:
            continue
        payload = _base_payload(
            left,
            rq="SYNTH-PERSONALITY",
            contrast="true_synthetic_ocean_vs_shuffled_profile",
        )
        payload["left_condition_id"] = "C3"
        payload["right_condition_id"] = "C4"
        payload["left_invalid_rate"] = float(left["invalid_rate"])
        payload["right_invalid_rate"] = float(right["invalid_rate"])
        payload["invalid_rate_difference"] = (
            payload["left_invalid_rate"] - payload["right_invalid_rate"]
        )
        for metric in UTILITY_METRICS:
            payload[f"left_{metric}"] = float(left[metric])
            payload[f"right_{metric}"] = float(right[metric])
            payload[f"delta_{metric}"] = float(left[metric]) - float(right[metric])
        payload.update(_matched_behavioral_diagnostics(left, right))
        output.append(payload)
    return output
