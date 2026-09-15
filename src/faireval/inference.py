from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .stats import (
    holm_adjust,
    paired_bootstrap_difference,
    paired_permutation_test,
    paired_wilcoxon_sensitivity,
)


def summarize_paired_estimands(
    pairs: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = ("ndcg", "recall"),
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2027,
    permutation_exact_max_n: int = 18,
    permutation_samples: int = 100_000,
    permutation_seed: int = 2027,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    """Summarize user-level paired estimands with the frozen FairEval inference stack.

    A hypothesis is a dataset x model-family x named contrast x optional
    intervention attribute. Holm correction is then applied across those
    pre-registered hypotheses within each RQ x metric x contrast family.

    The function rejects duplicate users within one hypothesis instead of
    treating multiple counterfactual alternatives from one user as independent
    observations. If a future confirmatory design has multiple alternatives per
    user, it must define a user-level reduction or hierarchical model explicitly.
    """
    if not pairs:
        return []
    if not metrics:
        raise ValueError("metrics cannot be empty")

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        key = (
            str(row["rq"]),
            str(row["contrast"]),
            str(row["dataset"]),
            str(row["model_family"]),
            str(row["requested_model_id"]),
            None if row.get("attribute") is None else str(row.get("attribute")),
        )
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        rq, contrast, dataset, family, model_id, attribute = key
        user_ids = [str(row["user_id"]) for row in rows]
        if len(set(user_ids)) != len(user_ids):
            raise ValueError(
                "duplicate user in one confirmatory hypothesis; define a user-level "
                f"reduction before inference: rq={rq}, dataset={dataset}, model={family}, "
                f"contrast={contrast}, attribute={attribute}"
            )

        for metric in metrics:
            left_field = f"left_{metric}"
            right_field = f"right_{metric}"
            if any(left_field not in row or right_field not in row for row in rows):
                raise ValueError(f"paired rows do not contain metric {metric!r}")
            left = [float(row[left_field]) for row in rows]
            right = [float(row[right_field]) for row in rows]

            bootstrap = paired_bootstrap_difference(
                left,
                right,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=bootstrap_seed,
            )
            permutation = paired_permutation_test(
                left,
                right,
                exact_max_n=permutation_exact_max_n,
                samples=permutation_samples,
                seed=permutation_seed,
            )
            wilcoxon = paired_wilcoxon_sensitivity(left, right)

            summaries.append(
                {
                    "schema_version": "faireval-inference-v1",
                    "rq": rq,
                    "contrast": contrast,
                    "metric": metric,
                    "dataset": dataset,
                    "model_family": family,
                    "requested_model_id": model_id,
                    "attribute": attribute,
                    "n_users": bootstrap.n,
                    "mean_paired_difference": bootstrap.mean_difference,
                    "median_paired_difference": bootstrap.median_difference,
                    "bootstrap_ci_low": bootstrap.ci_low,
                    "bootstrap_ci_high": bootstrap.ci_high,
                    "confidence": confidence,
                    "paired_permutation_p": permutation.p_value,
                    "paired_permutation_method": permutation.method,
                    "paired_permutation_count": permutation.permutations,
                    "wilcoxon_sensitivity_p": wilcoxon.p_value,
                    "wilcoxon_statistic": wilcoxon.statistic,
                    "wilcoxon_nonzero_pairs": wilcoxon.n_nonzero,
                    "matched_rank_biserial": wilcoxon.rank_biserial,
                    "left_invalid_rate_mean": sum(
                        float(row["left_invalid_rate"]) for row in rows
                    )
                    / len(rows),
                    "right_invalid_rate_mean": sum(
                        float(row["right_invalid_rate"]) for row in rows
                    )
                    / len(rows),
                    "invalid_rate_difference_mean": sum(
                        float(row["invalid_rate_difference"]) for row in rows
                    )
                    / len(rows),
                    "holm_adjusted_p": None,
                    "holm_family": f"{rq}|{metric}|{contrast}",
                }
            )

    # Freeze the multiple-comparison family before looking at the p-values:
    # RQ x metric x named contrast. All dataset/model/attribute hypotheses inside
    # that family are corrected jointly.
    by_family: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(summaries):
        by_family[(row["rq"], row["metric"], row["contrast"])].append(index)
    for indices in by_family.values():
        adjusted = holm_adjust([summaries[index]["paired_permutation_p"] for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            summaries[index]["holm_adjusted_p"] = value

    return sorted(
        summaries,
        key=lambda row: (
            row["rq"],
            row["contrast"],
            row["metric"],
            row["dataset"],
            row["model_family"],
            "" if row["attribute"] is None else row["attribute"],
        ),
    )
