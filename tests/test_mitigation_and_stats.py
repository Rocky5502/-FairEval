import math

import pytest

from faireval.mitigation import pair_rerank
from faireval.stats import (
    holm_adjust,
    paired_bootstrap_difference,
    paired_permutation_test,
    paired_wilcoxon_sensitivity,
)


def test_pair_returns_unique_candidates():
    out = pair_rerank(
        neutral_ranking=["a", "b", "c", "d"],
        counterfactual_rankings={
            "group_1": ["a", "b", "d", "c"],
            "group_2": ["a", "c", "b", "d"],
        },
        candidate_ids=["a", "b", "c", "d"],
        k=3,
        lambda_instability=0.2,
    )
    assert len(out) == 3
    assert len(set(out)) == 3
    assert set(out).issubset({"a", "b", "c", "d"})


def test_pair_keeps_consensus_top_item():
    out = pair_rerank(
        neutral_ranking=["a", "b", "c"],
        counterfactual_rankings={
            "g1": ["a", "c", "b"],
            "g2": ["a", "b", "c"],
        },
        candidate_ids=["a", "b", "c"],
        k=1,
        lambda_instability=1.0,
    )
    assert out == ["a"]


def test_pair_requires_at_least_two_identity_conditioned_rankings():
    with pytest.raises(ValueError, match="at least two"):
        pair_rerank(
            neutral_ranking=["a", "b", "c"],
            counterfactual_rankings={"only_group": ["a", "c", "b"]},
            candidate_ids=["a", "b", "c"],
            k=2,
            lambda_instability=0.2,
        )


def test_pair_rejects_out_of_candidate_and_duplicate_rankings():
    with pytest.raises(ValueError, match="outside candidate set"):
        pair_rerank(
            neutral_ranking=["a", "b", "c"],
            counterfactual_rankings={
                "g1": ["a", "x", "b"],
                "g2": ["a", "b", "c"],
            },
            candidate_ids=["a", "b", "c"],
            k=2,
            lambda_instability=0.2,
        )

    with pytest.raises(ValueError, match="duplicate"):
        pair_rerank(
            neutral_ranking=["a", "a", "c"],
            counterfactual_rankings={
                "g1": ["a", "b", "c"],
                "g2": ["a", "c", "b"],
            },
            candidate_ids=["a", "b", "c"],
            k=2,
            lambda_instability=0.2,
        )


def test_paired_bootstrap_preserves_positive_effect():
    summary = paired_bootstrap_difference(
        [0.8, 0.7, 0.9, 0.6],
        [0.5, 0.5, 0.6, 0.4],
        samples=2000,
        seed=7,
    )
    assert summary.n == 4
    assert summary.mean_difference > 0
    assert summary.ci_low > 0


def test_exact_paired_permutation_matches_enumerated_sign_flips():
    result = paired_permutation_test([2.0, 3.0], [0.0, 0.0])
    assert result.method == "exact_sign_flip"
    assert result.permutations == 4
    assert result.observed_mean_difference == 2.5
    assert result.p_value == pytest.approx(0.5)

    greater = paired_permutation_test(
        [2.0, 3.0],
        [0.0, 0.0],
        alternative="greater",
    )
    assert greater.p_value == pytest.approx(0.25)


def test_paired_permutation_all_zero_is_one():
    result = paired_permutation_test([1.0, 2.0], [1.0, 2.0])
    assert result.method == "exact_all_zero"
    assert result.p_value == 1.0


def test_monte_carlo_paired_permutation_is_seed_reproducible():
    left = [float(i % 5) for i in range(25)]
    right = [float((i + 1) % 5) for i in range(25)]
    a = paired_permutation_test(left, right, exact_max_n=2, samples=5000, seed=123)
    b = paired_permutation_test(left, right, exact_max_n=2, samples=5000, seed=123)
    assert a.method == "monte_carlo_sign_flip"
    assert a.p_value == b.p_value
    assert 0.0 < a.p_value <= 1.0


def test_wilcoxon_sensitivity_reports_directional_rank_biserial():
    result = paired_wilcoxon_sensitivity(
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [0.3, 0.4, 0.4, 0.2, 0.4],
    )
    assert result.n == 5
    assert result.n_nonzero == 5
    assert 0.0 <= result.p_value <= 1.0
    assert result.rank_biserial == pytest.approx(1.0)
    assert result.method == "scipy_wilcoxon_signed_rank"


def test_wilcoxon_all_zero_is_well_defined():
    result = paired_wilcoxon_sensitivity([1.0, 2.0], [1.0, 2.0])
    assert result.p_value == 1.0
    assert result.rank_biserial == 0.0
    assert result.n_nonzero == 0


def test_holm_adjust_is_monotone_and_restores_input_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_stats_reject_nonfinite_values_and_bad_probabilities():
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_difference([1.0, math.nan], [0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        paired_permutation_test([1.0, math.inf], [0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        paired_wilcoxon_sensitivity([1.0, math.nan], [0.0, 0.0])
    with pytest.raises(ValueError, match="probabilities"):
        holm_adjust([0.1, 1.2])
