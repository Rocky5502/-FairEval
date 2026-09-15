import pytest

from faireval.inference import summarize_paired_estimands


def _pair(
    user: str,
    family: str,
    left: float,
    right: float,
    *,
    left_invalid: float = 0.0,
    right_invalid: float = 0.0,
):
    return {
        "rq": "RQ1",
        "contrast": "observed_vs_demographic_counterfactual",
        "metric": "unused",
        "dataset": "movielens_1m",
        "user_id": user,
        "model_family": family,
        "requested_model_id": f"{family}-model",
        "attribute": "gender",
        "left_ndcg": left,
        "right_ndcg": right,
        "left_recall": left,
        "right_recall": right,
        "left_invalid_rate": left_invalid,
        "right_invalid_rate": right_invalid,
        "invalid_rate_difference": left_invalid - right_invalid,
    }


def test_inference_summarizes_at_user_level_and_applies_holm_family():
    pairs = []
    for family in ("openai", "anthropic"):
        pairs.extend(
            [
                _pair("u1", family, 0.8, 0.4),
                _pair("u2", family, 0.7, 0.4),
                _pair("u3", family, 0.9, 0.5),
                _pair("u4", family, 0.6, 0.3),
            ]
        )

    summaries = summarize_paired_estimands(
        pairs,
        metrics=("ndcg",),
        bootstrap_samples=1000,
        permutation_samples=1000,
    )
    assert len(summaries) == 2
    assert {row["model_family"] for row in summaries} == {"openai", "anthropic"}
    for row in summaries:
        assert row["n_users"] == 4
        assert row["mean_paired_difference"] > 0
        assert row["matched_rank_biserial"] == pytest.approx(1.0)
        assert row["holm_adjusted_p"] >= row["paired_permutation_p"]
        assert row["holm_family"] == "RQ1|ndcg|observed_vs_demographic_counterfactual"


def test_invalid_output_disparity_can_be_inferred_as_paired_user_outcome():
    pairs = [
        _pair("u1", "openai", 0.8, 0.4, left_invalid=0.0, right_invalid=1.0 / 3.0),
        _pair("u2", "openai", 0.7, 0.4, left_invalid=0.0, right_invalid=2.0 / 3.0),
        _pair("u3", "openai", 0.9, 0.5, left_invalid=0.0, right_invalid=1.0 / 3.0),
        _pair("u4", "openai", 0.6, 0.3, left_invalid=0.0, right_invalid=1.0),
    ]
    summaries = summarize_paired_estimands(
        pairs,
        metrics=("invalid_rate",),
        bootstrap_samples=1000,
        permutation_samples=1000,
    )
    assert len(summaries) == 1
    row = summaries[0]
    assert row["metric"] == "invalid_rate"
    assert row["n_users"] == 4
    assert row["mean_paired_difference"] < 0
    assert row["holm_family"] == (
        "RQ1|invalid_rate|observed_vs_demographic_counterfactual"
    )


def test_inference_rejects_duplicate_user_within_hypothesis():
    pairs = [
        _pair("u1", "openai", 0.8, 0.4),
        _pair("u1", "openai", 0.7, 0.3),
    ]
    with pytest.raises(ValueError, match="duplicate user"):
        summarize_paired_estimands(pairs, metrics=("ndcg",), bootstrap_samples=100)
