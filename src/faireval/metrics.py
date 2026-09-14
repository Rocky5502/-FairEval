from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence


def _top(ranking: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    return list(ranking[:k])


def ndcg_at_k(
    ranking: Sequence[str],
    relevant: Iterable[str] | Mapping[str, float],
    k: int,
) -> float:
    """Binary or graded nDCG@k."""
    if isinstance(relevant, Mapping):
        gains = {str(i): float(v) for i, v in relevant.items()}
    else:
        gains = {str(i): 1.0 for i in relevant}

    def dcg(items: Sequence[str]) -> float:
        return sum(gains.get(str(item), 0.0) / math.log2(rank + 2) for rank, item in enumerate(items))

    observed = dcg(_top(ranking, k))
    ideal_gains = sorted(gains.values(), reverse=True)[:k]
    ideal = sum(g / math.log2(rank + 2) for rank, g in enumerate(ideal_gains))
    return 0.0 if ideal <= 0 else observed / ideal


def recall_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = {str(x) for x in relevant}
    if not relevant_set:
        return 0.0
    hits = len(set(_top(ranking, k)) & relevant_set)
    return hits / len(relevant_set)


def mrr_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = {str(x) for x in relevant}
    for idx, item in enumerate(_top(ranking, k), start=1):
        if str(item) in relevant_set:
            return 1.0 / idx
    return 0.0


def jaccard_at_k(left: Sequence[str], right: Sequence[str], k: int) -> float:
    a, b = set(_top(left, k)), set(_top(right, k))
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def rbo_at_k(left: Sequence[str], right: Sequence[str], k: int, p: float = 0.9) -> float:
    """Normalized finite-depth Rank-Biased Overlap diagnostic.

    This finite-depth form is normalized by the cumulative depth weights so it
    stays in [0, 1] and is suitable for matched top-k lists. It is deliberately
    a *behavioral shift* metric, not a fairness definition.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    left_k, right_k = _top(left, k), _top(right, k)
    max_depth = min(k, max(len(left_k), len(right_k)))
    if max_depth == 0:
        return 1.0

    weighted_overlap = 0.0
    weight_sum = 0.0
    for depth in range(1, max_depth + 1):
        overlap = len(set(left_k[:depth]) & set(right_k[:depth])) / depth
        weight = (1.0 - p) * (p ** (depth - 1))
        weighted_overlap += weight * overlap
        weight_sum += weight
    return weighted_overlap / weight_sum


def counterfactual_utility_gap(utility_a: float, utility_b: float) -> float:
    """Signed paired utility gap U(a)-U(a')."""
    return float(utility_a) - float(utility_b)


def personality_value_added(true_utility: float, shuffled_utility: float) -> float:
    """Matched benefit of measured personality over shuffled-personality control."""
    return float(true_utility) - float(shuffled_utility)


def group_utility_disparity(group_utilities: Mapping[str, Sequence[float]]) -> float:
    """Range of group mean utility; report group CIs alongside this scalar."""
    means = []
    for values in group_utilities.values():
        if values:
            means.append(sum(float(v) for v in values) / len(values))
    if len(means) < 2:
        return 0.0
    return max(means) - min(means)


def discounted_exposure(
    ranking: Sequence[str],
    item_groups: Mapping[str, str],
    k: int,
) -> dict[str, float]:
    """Normalized position-discounted exposure by an auditable item group."""
    exposure: dict[str, float] = defaultdict(float)
    total = 0.0
    for idx, item_id in enumerate(_top(ranking, k), start=1):
        group = item_groups.get(str(item_id))
        if group is None:
            continue
        weight = 1.0 / math.log2(idx + 1)
        exposure[str(group)] += weight
        total += weight
    if total:
        return {group: value / total for group, value in exposure.items()}
    return {}


def counterfactual_exposure_gap(
    left_exposure: Mapping[str, float], right_exposure: Mapping[str, float]
) -> float:
    """Total-variation distance between exposure distributions."""
    groups = set(left_exposure) | set(right_exposure)
    return 0.5 * sum(abs(left_exposure.get(g, 0.0) - right_exposure.get(g, 0.0)) for g in groups)
