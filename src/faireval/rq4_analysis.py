from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .freeze import load_frozen_instances
from .metrics import ndcg_at_k, recall_at_k
from .mitigation import alpha_grid, lambda_grid, pair_contextual_rerank
from .schema import UserInstance


RQ4_DATASETS = {"movielens_1m", "lastfm_1k"}


def _stable_user_digest(dataset: str, user_id: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}|rq4|{dataset}|{user_id}".encode("utf-8")).hexdigest()


def _validation_user_keys(
    triplets: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    fraction: float,
) -> set[tuple[str, str]]:
    """Select an exact deterministic validation subset independently per dataset.

    Users, not repetitions/models, are the split unit. Every model and repetition
    for one user therefore remains on the same side of the validation/test wall.
    Within each dataset we stable-hash users, sort by that hash, and take exactly
    ``round(fraction * n_users)`` (bounded to leave at least one test user).
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must be in (0,1)")
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for triplet in triplets:
        observed = triplet["observed"]
        by_dataset[str(observed["dataset"])].add(str(observed["user_id"]))

    selected: set[tuple[str, str]] = set()
    for dataset, user_ids in sorted(by_dataset.items()):
        if len(user_ids) < 2:
            raise ValueError(f"RQ4 dataset {dataset!r} needs at least two users for validation/test")
        ordered = sorted(
            user_ids,
            key=lambda user_id: (_stable_user_digest(dataset, user_id, seed=seed), user_id),
        )
        n_validation = max(1, int(round(len(ordered) * fraction)))
        n_validation = min(n_validation, len(ordered) - 1)
        selected.update((dataset, user_id) for user_id in ordered[:n_validation])
    return selected


def _instance_index(freeze_root: Path, datasets: Iterable[str]) -> dict[tuple[str, str], UserInstance]:
    index: dict[tuple[str, str], UserInstance] = {}
    for dataset in sorted(set(datasets)):
        for instance in load_frozen_instances(freeze_root / dataset):
            instance.validate()
            key = (dataset, str(instance.user_id))
            if key in index:
                raise ValueError(f"duplicate frozen instance {key!r}")
            index[key] = instance
    return index


def _base_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["dataset"]),
        str(row["user_id"]),
        str(row["model_family"]),
        str(row["requested_model_id"]),
        str(row["prompt_template_id"]),
        str(row["prompt_mode"]),
        str(row["cue_id"]),
        row.get("candidate_order_seed"),
        int(row["k"]),
        int(row["repetition"]),
        str(row["code_commit_sha"]),
        row.get("plan_sha256"),
    )


def _collect_triplets(scored_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collect C0/C1/primary-C2 rows for one user/model/repetition.

    Only rows explicitly tagged ``rq4_mitigation_baseline`` are eligible. This
    excludes age robustness and other demographic interventions from the frozen
    RQ4 operating-point search.
    """
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        if str(row.get("dataset")) not in RQ4_DATASETS:
            continue
        if "rq4_mitigation_baseline" not in set(row.get("analysis_roles", [])):
            continue
        cid = str(row.get("condition_id", ""))
        if cid == "C0" or cid == "C1" or cid.startswith("C2:"):
            grouped[_base_key(row)].append(row)

    triplets: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        neutral = [row for row in rows if str(row["condition_id"]) == "C0"]
        observed = [row for row in rows if str(row["condition_id"]) == "C1"]
        counterfactual = [row for row in rows if str(row["condition_id"]).startswith("C2:")]
        if len(neutral) != 1 or len(observed) != 1 or len(counterfactual) != 1:
            raise ValueError(
                "RQ4 baseline requires exactly one C0, one C1 and one primary C2 "
                f"per user/model/repetition; key={key!r}, counts="
                f"{(len(neutral), len(observed), len(counterfactual))}"
            )
        intervention = counterfactual[0].get("condition_intervention", {})
        if not isinstance(intervention, Mapping) or intervention.get("attribute") != "gender":
            raise ValueError("RQ4 primary C2 must be the preregistered gender intervention")
        triplets.append(
            {
                "key": key,
                "neutral": neutral[0],
                "observed": observed[0],
                "counterfactual": counterfactual[0],
            }
        )
    return triplets


def evaluate_pair_grid_point(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    freeze_root: Path,
    alpha: float,
    lambda_instability: float,
    split_seed: int = 2027,
    validation_fraction: float = 0.20,
) -> list[dict[str, Any]]:
    """Evaluate one contextual-PAIR point without selecting on test users.

    PAIR can only post-process a repetition when C0/C1/C2 all produced valid
    rankings. Unavailable repetitions contribute zero end-to-end PAIR utility and
    are reported through ``pair_available``; CUG is summarized only when both
    contextual rerankings are defined. This prevents a missing base ranking from
    being mislabeled as a fair zero-gap recommendation.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0,1)")
    triplets = _collect_triplets(scored_rows)
    if not triplets:
        raise ValueError("no RQ4 C0/C1/C2 baseline triplets found")
    validation_users = _validation_user_keys(
        triplets,
        seed=split_seed,
        fraction=validation_fraction,
    )
    datasets = {str(t["observed"]["dataset"]) for t in triplets}
    instances = _instance_index(freeze_root, datasets)

    rows: list[dict[str, Any]] = []
    for triplet in triplets:
        neutral = triplet["neutral"]
        observed = triplet["observed"]
        counterfactual = triplet["counterfactual"]
        dataset = str(observed["dataset"])
        user_id = str(observed["user_id"])
        instance = instances[(dataset, user_id)]
        k = int(observed["k"])
        if int(neutral["k"]) != k or int(counterfactual["k"]) != k:
            raise ValueError("RQ4 triplet disagrees on ranking cutoff")

        baseline_utility = (float(observed["ndcg"]) + float(counterfactual["ndcg"])) / 2.0
        baseline_abs_cug = abs(float(observed["ndcg"]) - float(counterfactual["ndcg"]))
        available = bool(
            neutral["final_valid"]
            and observed["final_valid"]
            and counterfactual["final_valid"]
        )

        pair_observed_ndcg: float | None = None
        pair_counterfactual_ndcg: float | None = None
        pair_observed_recall: float | None = None
        pair_counterfactual_recall: float | None = None
        pair_abs_cug: float | None = None
        pair_system_utility = 0.0

        if available:
            identity_rankings = {
                "observed": list(observed["ranking"]),
                "counterfactual": list(counterfactual["ranking"]),
            }
            candidates = instance.candidate_ids()
            observed_rank = pair_contextual_rerank(
                neutral_ranking=list(neutral["ranking"]),
                identity_rankings=identity_rankings,
                focal_context="observed",
                candidate_ids=candidates,
                k=k,
                alpha_neutral=alpha,
                lambda_instability=lambda_instability,
            )
            counterfactual_rank = pair_contextual_rerank(
                neutral_ranking=list(neutral["ranking"]),
                identity_rankings=identity_rankings,
                focal_context="counterfactual",
                candidate_ids=candidates,
                k=k,
                alpha_neutral=alpha,
                lambda_instability=lambda_instability,
            )
            relevant = instance.relevant_item_ids
            pair_observed_ndcg = ndcg_at_k(observed_rank, relevant, k)
            pair_counterfactual_ndcg = ndcg_at_k(counterfactual_rank, relevant, k)
            pair_observed_recall = recall_at_k(observed_rank, relevant, k)
            pair_counterfactual_recall = recall_at_k(counterfactual_rank, relevant, k)
            pair_abs_cug = abs(pair_observed_ndcg - pair_counterfactual_ndcg)
            pair_system_utility = (pair_observed_ndcg + pair_counterfactual_ndcg) / 2.0

        split = "validation" if (dataset, user_id) in validation_users else "test"
        rows.append(
            {
                "schema_version": "faireval-rq4-pair-run-v1",
                "dataset": dataset,
                "user_id": user_id,
                "model_family": str(observed["model_family"]),
                "requested_model_id": str(observed["requested_model_id"]),
                "repetition": int(observed["repetition"]),
                "k": k,
                "split": split,
                "split_method": "exact_hash_ranked_per_dataset_user",
                "alpha": float(alpha),
                "lambda_instability": float(lambda_instability),
                "pair_available": available,
                "baseline_identity_ndcg_mean": baseline_utility,
                "baseline_abs_cug_ndcg": baseline_abs_cug,
                "pair_identity_ndcg_mean_system": pair_system_utility,
                "pair_observed_ndcg": pair_observed_ndcg,
                "pair_counterfactual_ndcg": pair_counterfactual_ndcg,
                "pair_observed_recall": pair_observed_recall,
                "pair_counterfactual_recall": pair_counterfactual_recall,
                "pair_abs_cug_ndcg": pair_abs_cug,
                "code_commit_sha": str(observed["code_commit_sha"]),
                "plan_sha256": observed.get("plan_sha256"),
            }
        )
    return rows


def summarize_grid_point(rows: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, Any]:
    chosen = [row for row in rows if str(row["split"]) == split]
    if not chosen:
        raise ValueError(f"RQ4 split {split!r} is empty")
    baseline_utility = sum(float(row["baseline_identity_ndcg_mean"]) for row in chosen) / len(chosen)
    pair_utility = sum(float(row["pair_identity_ndcg_mean_system"]) for row in chosen) / len(chosen)
    available = [row for row in chosen if bool(row["pair_available"])]
    coverage = len(available) / len(chosen)
    pair_abs_cug = (
        None
        if not available
        else sum(float(row["pair_abs_cug_ndcg"]) for row in available) / len(available)
    )
    baseline_abs_cug = (
        None
        if not available
        else sum(float(row["baseline_abs_cug_ndcg"]) for row in available) / len(available)
    )
    retention = 1.0 if baseline_utility == 0.0 and pair_utility == 0.0 else (
        0.0 if baseline_utility == 0.0 else pair_utility / baseline_utility
    )
    first = chosen[0]
    return {
        "schema_version": "faireval-rq4-grid-summary-v1",
        "split": split,
        "alpha": float(first["alpha"]),
        "lambda_instability": float(first["lambda_instability"]),
        "n_repetition_rows": len(chosen),
        "n_users": len({(str(row["dataset"]), str(row["user_id"])) for row in chosen}),
        "pair_availability": coverage,
        "baseline_identity_ndcg_mean": baseline_utility,
        "pair_identity_ndcg_mean_system": pair_utility,
        "utility_retention": retention,
        "baseline_abs_cug_ndcg_on_available": baseline_abs_cug,
        "pair_abs_cug_ndcg_on_available": pair_abs_cug,
    }


def select_operating_point(
    validation_summaries: Sequence[Mapping[str, Any]],
    *,
    utility_floor_ratio: float = 0.95,
) -> dict[str, Any]:
    """Freeze one global PAIR operating point using validation evidence only."""
    if not 0.0 < utility_floor_ratio <= 1.0:
        raise ValueError("utility_floor_ratio must be in (0,1]")
    eligible = [
        row
        for row in validation_summaries
        if row.get("pair_abs_cug_ndcg_on_available") is not None
        and float(row["utility_retention"]) >= utility_floor_ratio
    ]
    if not eligible:
        raise ValueError(
            "no PAIR grid point satisfies the preregistered validation utility floor; "
            "do not relax the threshold after observing test outcomes"
        )
    selected = min(
        eligible,
        key=lambda row: (
            float(row["pair_abs_cug_ndcg_on_available"]),
            -float(row["pair_identity_ndcg_mean_system"]),
            float(row["lambda_instability"]),
            float(row["alpha"]),
        ),
    )
    return {
        "schema_version": "faireval-rq4-operating-point-v1",
        "alpha": float(selected["alpha"]),
        "lambda_instability": float(selected["lambda_instability"]),
        "utility_floor_ratio": float(utility_floor_ratio),
        "selection_objective": "min_abs_cug_subject_to_validation_utility_retention",
        "tie_break": "max_pair_utility_then_lower_lambda_then_lower_alpha",
        "validation_summary": dict(selected),
    }


def build_rq4_pair_artifact(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    freeze_root: Path,
    split_seed: int = 2027,
    validation_fraction: float = 0.20,
    utility_floor_ratio: float = 0.95,
    alphas: Sequence[float] | None = None,
    lambdas: Sequence[float] | None = None,
) -> dict[str, Any]:
    alpha_values = tuple(alpha_grid() if alphas is None else alphas)
    lambda_values = tuple(lambda_grid() if lambdas is None else lambdas)
    if not alpha_values or not lambda_values:
        raise ValueError("RQ4 alpha/lambda grids cannot be empty")

    validation_frontier: list[dict[str, Any]] = []
    by_point: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for alpha in alpha_values:
        for lam in lambda_values:
            rows = evaluate_pair_grid_point(
                scored_rows,
                freeze_root=freeze_root,
                alpha=float(alpha),
                lambda_instability=float(lam),
                split_seed=split_seed,
                validation_fraction=validation_fraction,
            )
            by_point[(float(alpha), float(lam))] = rows
            validation_frontier.append(summarize_grid_point(rows, split="validation"))

    operating_point = select_operating_point(
        validation_frontier,
        utility_floor_ratio=utility_floor_ratio,
    )
    selected_rows = by_point[
        (operating_point["alpha"], operating_point["lambda_instability"])
    ]
    test_summary = summarize_grid_point(selected_rows, split="test")

    validation_user_keys = {
        (str(row["dataset"]), str(row["user_id"]))
        for row in selected_rows
        if row["split"] == "validation"
    }
    test_user_keys = {
        (str(row["dataset"]), str(row["user_id"]))
        for row in selected_rows
        if row["split"] == "test"
    }

    return {
        "schema_version": "faireval-rq4-pair-artifact-v1",
        "split_seed": int(split_seed),
        "validation_fraction": float(validation_fraction),
        "split_method": "exact_hash_ranked_per_dataset_user",
        "validation_users": len(validation_user_keys),
        "test_users": len(test_user_keys),
        "utility_floor_ratio": float(utility_floor_ratio),
        "alpha_grid": [float(value) for value in alpha_values],
        "lambda_grid": [float(value) for value in lambda_values],
        "operating_point": operating_point,
        "validation_frontier": validation_frontier,
        "test_summary": test_summary,
        "selected_test_rows": [dict(row) for row in selected_rows if row["split"] == "test"],
        "selection_used_test_outcomes": False,
        "per_model_or_dataset_tuning": False,
    }
