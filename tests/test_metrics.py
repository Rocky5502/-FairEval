import math

from faireval.metrics import (
    counterfactual_exposure_gap,
    counterfactual_utility_gap,
    discounted_exposure,
    jaccard_at_k,
    mrr_at_k,
    ndcg_at_k,
    personality_value_added,
    rbo_at_k,
    recall_at_k,
)


def test_perfect_binary_ranking():
    ranking = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert math.isclose(ndcg_at_k(ranking, relevant, 2), 1.0)
    assert math.isclose(recall_at_k(ranking, relevant, 2), 1.0)
    assert math.isclose(mrr_at_k(ranking, relevant, 3), 1.0)


def test_behavioral_similarity_bounds():
    left = ["a", "b", "c"]
    same = ["a", "b", "c"]
    disjoint = ["x", "y", "z"]
    assert jaccard_at_k(left, same, 3) == 1.0
    assert jaccard_at_k(left, disjoint, 3) == 0.0
    assert math.isclose(rbo_at_k(left, same, 3), 1.0)
    assert math.isclose(rbo_at_k(left, disjoint, 3), 0.0)


def test_paired_effects_keep_sign():
    assert math.isclose(counterfactual_utility_gap(0.7, 0.5), 0.2)
    assert math.isclose(personality_value_added(0.8, 0.6), 0.2)


def test_exposure_gap():
    groups = {"a": "g1", "b": "g1", "c": "g2"}
    left = discounted_exposure(["a", "b", "c"], groups, 3)
    right = discounted_exposure(["c", "a", "b"], groups, 3)
    gap = counterfactual_exposure_gap(left, right)
    assert 0.0 <= gap <= 1.0
    assert gap > 0.0
