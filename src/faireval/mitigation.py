from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _rank_map(ranking: Sequence[str], candidates: Sequence[str]) -> dict[str, int]:
    default_rank = len(candidates) + 1
    positions = {str(item): idx for idx, item in enumerate(ranking, start=1)}
    return {str(item): positions.get(str(item), default_rank) for item in candidates}


def pair_rerank(
    *,
    neutral_ranking: Sequence[str],
    counterfactual_rankings: Mapping[str, Sequence[str]],
    candidate_ids: Sequence[str],
    k: int,
    lambda_instability: float,
) -> list[str]:
    """Preference-Aligned Identity Re-ranking (PAIR) reference implementation.

    ``relevance_consensus`` uses reciprocal-rank support from the neutral ranking
    and all counterfactual rankings. ``counterfactual_instability`` is the
    normalized rank range across counterfactual identities. This deliberately
    simple method is an auditable baseline; the paper must not claim superiority
    until validation/test results support it.
    """
    if k <= 0 or k > len(candidate_ids):
        raise ValueError("invalid k")
    if lambda_instability < 0:
        raise ValueError("lambda_instability must be non-negative")
    if not counterfactual_rankings:
        raise ValueError("at least one counterfactual ranking is required")

    candidate_ids = [str(x) for x in candidate_ids]
    rankings = {"neutral": neutral_ranking, **dict(counterfactual_rankings)}
    rank_maps = {name: _rank_map(ranking, candidate_ids) for name, ranking in rankings.items()}
    cf_maps = [rank_maps[name] for name in counterfactual_rankings]
    denom = max(1, len(candidate_ids) - 1)

    scores: dict[str, float] = {}
    for item in candidate_ids:
        reciprocal_support = [1.0 / rank_map[item] for rank_map in rank_maps.values()]
        relevance_consensus = sum(reciprocal_support) / len(reciprocal_support)

        cf_positions = [rank_map[item] for rank_map in cf_maps]
        if len(cf_positions) <= 1:
            instability = 0.0
        else:
            instability = (max(cf_positions) - min(cf_positions)) / denom

        scores[item] = relevance_consensus - lambda_instability * instability

    # Stable deterministic tie-break: neutral rank first, then item ID.
    neutral_positions = rank_maps["neutral"]
    return sorted(
        candidate_ids,
        key=lambda item: (-scores[item], neutral_positions[item], item),
    )[:k]


def lambda_grid() -> tuple[float, ...]:
    """Pre-declared validation grid for Pareto analysis."""
    return (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
