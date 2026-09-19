import pytest

from faireval.inference import summarize_paired_estimands
from faireval.trait_analysis import build_rq2_one_trait_pairs


def _row(condition_id: str, *, ndcg: float, intervention=None):
    return {
        "dataset": "personality2018",
        "user_id": "u1",
        "model_family": "openai",
        "requested_model_id": "gpt-5.6-terra",
        "condition_id": condition_id,
        "condition_intervention": intervention or {},
        "prompt_template_id": "field_v2_a",
        "prompt_mode": "audit",
        "cue_id": "structured_key_value",
        "candidate_order_seed": 7,
        "k": 10,
        "invalid_rate": 0.0,
        "ndcg": ndcg,
        "recall": ndcg,
        "mrr": ndcg,
        "valid_only_ndcg": ndcg,
        "valid_only_recall": ndcg,
        "valid_only_mrr": ndcg,
        "ranking_by_repetition": [],
        "code_commit_sha": "abc",
        "plan_sha256": "plan",
    }


def test_build_rq2_one_trait_pairs_keeps_trait_as_hypothesis_attribute():
    rows = [
        _row("C3", ndcg=0.80),
        _row(
            "C5:openness",
            ndcg=0.65,
            intervention={
                "type": "one_trait_personality_counterfactual",
                "trait": "openness",
                "observed_value": 0.8,
                "counterfactual_value": 0.2,
                "donor_user_id": "u9",
                "other_traits_held_fixed": True,
            },
        ),
        _row(
            "C5:extraversion",
            ndcg=0.75,
            intervention={
                "type": "one_trait_personality_counterfactual",
                "trait": "extraversion",
                "observed_value": 0.4,
                "counterfactual_value": 0.9,
                "donor_user_id": "u8",
                "other_traits_held_fixed": True,
            },
        ),
    ]
    pairs = build_rq2_one_trait_pairs(rows)
    assert len(pairs) == 2
    by_trait = {row["attribute"]: row for row in pairs}
    assert by_trait["openness"]["delta_ndcg"] == pytest.approx(0.15)
    assert by_trait["extraversion"]["delta_ndcg"] == pytest.approx(0.05)
    assert all(row["contrast"] == "true_vs_one_trait_counterfactual" for row in pairs)
    assert all(row["robustness_only"] is True for row in pairs)


def test_trait_inference_does_not_mix_traits_into_one_hypothesis():
    pairs = build_rq2_one_trait_pairs(
        [
            _row("C3", ndcg=0.80),
            _row(
                "C5:openness",
                ndcg=0.65,
                intervention={
                    "type": "one_trait_personality_counterfactual",
                    "trait": "openness",
                    "observed_value": 0.8,
                    "counterfactual_value": 0.2,
                    "donor_user_id": "u9",
                    "other_traits_held_fixed": True,
                },
            ),
            _row(
                "C5:agreeableness",
                ndcg=0.70,
                intervention={
                    "type": "one_trait_personality_counterfactual",
                    "trait": "agreeableness",
                    "observed_value": 0.6,
                    "counterfactual_value": 0.1,
                    "donor_user_id": "u7",
                    "other_traits_held_fixed": True,
                },
            ),
        ]
    )
    summaries = summarize_paired_estimands(
        pairs,
        metrics=("ndcg",),
        bootstrap_samples=20,
        permutation_samples=20,
    )
    assert {row["attribute"] for row in summaries} == {"openness", "agreeableness"}
    assert all(row["holm_family"] == "RQ2|ndcg|true_vs_one_trait_counterfactual" for row in summaries)


def test_trait_pair_rejects_condition_intervention_mismatch():
    with pytest.raises(ValueError, match="condition/intervention mismatch"):
        build_rq2_one_trait_pairs(
            [
                _row("C3", ndcg=0.80),
                _row(
                    "C5:openness",
                    ndcg=0.70,
                    intervention={
                        "type": "one_trait_personality_counterfactual",
                        "trait": "extraversion",
                        "other_traits_held_fixed": True,
                    },
                ),
            ]
        )
