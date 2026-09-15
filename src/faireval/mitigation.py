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
    """Legacy context-averaged PAIR consensus score.

    This helper is retained for reproducibility of early development experiments,
    but it is **not** the primary ECIR RQ4 mitigation because averaging all
    identity-conditioned rankings yields one context-independent output and thus
    makes counterfactual output disparity degenerate to zero by construction.
    Use :func:`pair_contextual_scores` for the frozen RQ4 design.
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
    """Legacy context-averaged PAIR ranking; use contextual PAIR for RQ4."""
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
    return sorted(
        candidate_ids,
        key=lambda item: (-scores[item], neutral_positions[item], item),
    )[:k]


def pair_contextual_scores(
    *,
    neutral_ranking: Sequence[str],
    identity_rankings: Mapping[str, Sequence[str]],
    focal_context: str,
    candidate_ids: Sequence[str],
    alpha_neutral: float,
    lambda_instability: float,
) -> dict[str, float]:
    """Contextual PAIR score used by the ECIR RQ4 mitigation.

    For identity context ``a`` and item ``i``:

        fit_a(i) = alpha*b(i) + (1-alpha)*r_a(i)
        instability(i) = Var_{a' in A}[r_{a'}(i)]
        PAIR_a(i) = fit_a(i) - lambda*instability(i)

    ``b`` is preference-only support and ``r_a`` is support under the *focal*
    identity context. Unlike context averaging, this formulation preserves useful
    context-specific relevance when it is stable while progressively penalizing
    items whose support varies across counterfactual identities. Therefore the
    post-mitigation counterfactual gap is an empirical outcome, not forced to zero.
    """
    if not 0.0 <= alpha_neutral <= 1.0:
        raise ValueError("alpha_neutral must be in [0,1]")
    if lambda_instability < 0.0:
        raise ValueError("lambda_instability must be non-negative")
    if len(identity_rankings) < 2:
        raise ValueError("contextual PAIR requires at least two identity-conditioned rankings")
    if focal_context not in identity_rankings:
        raise ValueError(f"focal_context {focal_context!r} is not present in identity_rankings")

    candidates = [str(item) for item in candidate_ids]
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate_ids must be non-empty and unique")
    n_candidates = len(candidates)
    neutral = _validate_ranking(neutral_ranking, candidates, label="neutral_ranking")
    validated = {
        str(name): _validate_ranking(ranking, candidates, label=f"identity[{name}]")
        for name, ranking in identity_rankings.items()
    }

    neutral_map = _rank_map(neutral, candidates)
    rank_maps = {name: _rank_map(ranking, candidates) for name, ranking in validated.items()}
    focal_map = rank_maps[focal_context]
    scores: dict[str, float] = {}
    for item in candidates:
        neutral_support = _rank_support(neutral_map[item], n_candidates)
        focal_support = _rank_support(focal_map[item], n_candidates)
        across_contexts = [
            _rank_support(rank_map[item], n_candidates) for rank_map in rank_maps.values()
        ]
        fit = alpha_neutral * neutral_support + (1.0 - alpha_neutral) * focal_support
        instability = pvariance(across_contexts)
        scores[item] = fit - lambda_instability * instability
    return scores


def pair_contextual_rerank(
    *,
    neutral_ranking: Sequence[str],
    identity_rankings: Mapping[str, Sequence[str]],
    focal_context: str,
    candidate_ids: Sequence[str],
    k: int,
    lambda_instability: float,
    alpha_neutral: float = 0.5,
) -> list[str]:
    if k <= 0 or k > len(candidate_ids):
        raise ValueError("invalid k")
    candidates = [str(item) for item in candidate_ids]
    scores = pair_contextual_scores(
        neutral_ranking=neutral_ranking,
        identity_rankings=identity_rankings,
        focal_context=focal_context,
        candidate_ids=candidates,
        alpha_neutral=alpha_neutral,
        lambda_instability=lambda_instability,
    )
    focal_positions = _rank_map(identity_rankings[focal_context], candidates)
    neutral_positions = _rank_map(neutral_ranking, candidates)
    return sorted(
        candidates,
        key=lambda item: (-scores[item], focal_positions[item], neutral_positions[item], item),
    )[:k]


def alpha_grid() -> tuple[float, ...]:
    """Pre-declared validation grid for neutral-preference weight."""
    return (0.25, 0.5, 0.75)


def lambda_grid() -> tuple[float, ...]:
    """Pre-declared validation grid for counterfactual-instability penalty."""
    return (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
