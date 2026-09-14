from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import pvariance


def _validate_ranking(
    ranking: Sequence[str],
    candidates: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    values = tuple(str(item) for item in ranking)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate item IDs")
    candidate_set = {str(item) for item in candidates}
    unknown = sorted(set(values) - candidate_set)
    if unknown:
        raise ValueError(f"{label} contains items outside candidate set: {unknown!r}")
    return values


def _rank_map(ranking: Sequence[str], candidates: Sequence[str]) -> dict[str, int]:
    default_rank = len(candidates) + 1
    positions = {str(item): idx for idx, item in enumerate(ranking, start=1)}
    return {str(item): positions.get(str(item), default_rank) for item in candidates}


def _rank_support(rank: int, n_candidates: int) -> float:
    """Map rank 1..N to linear support 1..1/N; missing rank N+1 maps to 0."""
    if rank > n_candidates:
        return 0.0
    return (n_candidates + 1 - rank) / n_candidates


def pair_scores(
    *,
    neutral_ranking: Sequence[str],
    counterfactual_rankings: Mapping[str, Sequence[str]],
    candidate_ids: Sequence[str],
    alpha_neutral: float,
    lambda_instability: float,
) -> dict[str, float]:
    """Compute Preference-Aligned Identity Re-ranking (PAIR) scores.

    Let ``b(i)`` be normalized support in the preference-only/neutral ranking and
    ``r_a(i)`` support under identity context ``a``. PAIR uses

        consensus(i) = alpha*b(i) + (1-alpha)*mean_a r_a(i)
        instability(i) = Var_a[r_a(i)]
        score(i) = consensus(i) - lambda*instability(i)

    At least two identity-conditioned rankings are required: with only one,
    counterfactual instability is undefined for the purpose of this mitigation
    and the variance penalty degenerates to zero. ``alpha`` and ``lambda`` are
    validation-only hyperparameters; the test split must never choose them.
    """
    if not 0.0 <= alpha_neutral <= 1.0:
        raise ValueError("alpha_neutral must be in [0,1]")
    if lambda_instability < 0.0:
        raise ValueError("lambda_instability must be non-negative")
    if len(counterfactual_rankings) < 2:
        raise ValueError("PAIR requires at least two identity-conditioned rankings")

    candidate_ids = [str(x) for x in candidate_ids]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique")
    n_candidates = len(candidate_ids)
    if n_candidates == 0:
        raise ValueError("candidate_ids cannot be empty")

    neutral = _validate_ranking(neutral_ranking, candidate_ids, label="neutral_ranking")
    validated_cf = {
        str(name): _validate_ranking(ranking, candidate_ids, label=f"counterfactual[{name}]")
        for name, ranking in counterfactual_rankings.items()
    }

    neutral_map = _rank_map(neutral, candidate_ids)
    cf_maps = [_rank_map(ranking, candidate_ids) for ranking in validated_cf.values()]

    scores: dict[str, float] = {}
    for item in candidate_ids:
        neutral_support = _rank_support(neutral_map[item], n_candidates)
        cf_support = [_rank_support(rank_map[item], n_candidates) for rank_map in cf_maps]
        cf_mean = sum(cf_support) / len(cf_support)
        consensus = alpha_neutral * neutral_support + (1.0 - alpha_neutral) * cf_mean
        instability = pvariance(cf_support)
        scores[item] = consensus - lambda_instability * instability
    return scores


def pair_rerank(
    *,
    neutral_ranking: Sequence[str],
    counterfactual_rankings: Mapping[str, Sequence[str]],
    candidate_ids: Sequence[str],
    k: int,
    lambda_instability: float,
    alpha_neutral: float = 0.5,
) -> list[str]:
    """Return the top-K PAIR ranking using the frozen scoring definition."""
    if k <= 0 or k > len(candidate_ids):
        raise ValueError("invalid k")

    candidate_ids = [str(x) for x in candidate_ids]
    scores = pair_scores(
        neutral_ranking=neutral_ranking,
        counterfactual_rankings=counterfactual_rankings,
        candidate_ids=candidate_ids,
        alpha_neutral=alpha_neutral,
        lambda_instability=lambda_instability,
    )
    neutral_positions = _rank_map(neutral_ranking, candidate_ids)

    # Stable deterministic tie-break: neutral rank first, then item ID.
    return sorted(
        candidate_ids,
        key=lambda item: (-scores[item], neutral_positions[item], item),
    )[:k]


def alpha_grid() -> tuple[float, ...]:
    """Pre-declared validation grid for neutral-preference weight."""
    return (0.25, 0.5, 0.75)


def lambda_grid() -> tuple[float, ...]:
    """Pre-declared validation grid for counterfactual-instability penalty."""
    return (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
