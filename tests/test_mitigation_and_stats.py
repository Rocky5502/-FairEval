from faireval.mitigation import pair_rerank
from faireval.stats import paired_bootstrap_difference


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
