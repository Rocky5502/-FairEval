import pytest

from faireval.fairsynth_analysis import (
    build_fairsynth_identity_pairs,
    build_fairsynth_personality_pairs,
)


def _agg(condition_id: str, score: float, *, intervention=None, ranking=None):
    return {
        "schema_version": "faireval-user-condition-v1",
        "dataset": "fairsynth360",
        "user_id": "u001",
        "model_family": "qwen25_local",
        "requested_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "condition_id": condition_id,
        "condition_name": {
            "C1": "observed_demographic",
            "C3": "true_personality",
            "C4": "shuffled_personality",
        }.get(condition_id, "counterfactual_demographic"),
        "condition_intervention": intervention or {},
        "confirmatory": False,
        "analysis_roles": ["synthetic_identity_sanity"],
        "prompt_template_id": "field_v2_a",
        "prompt_mode": "audit",
        "cue_id": "structured_key_value",
        "candidate_order_seed": 123,
        "k": 3,
        "repetitions": 1,
        "ranking_by_repetition": [
            {
                "repetition": 0,
                "valid": ranking is not None,
                "ranking": [] if ranking is None else list(ranking),
            }
        ],
        "invalid_rate": 0.0 if ranking is not None else 1.0,
        "ndcg": score,
        "recall": score,
        "mrr": score,
        "valid_only_ndcg": score if ranking is not None else None,
        "valid_only_recall": score if ranking is not None else None,
        "valid_only_mrr": score if ranking is not None else None,
        "code_commit_sha": "abc",
        "plan_sha256": "plan",
    }


def test_identity_two_counterfactuals_reduce_to_one_user_level_pair():
    rows = [
        _agg("C1", 0.8, ranking=("i1", "i2", "i3")),
        _agg(
            "C2:synthetic_identity_group_A_to_B",
            0.6,
            intervention={
                "attribute": "synthetic_identity_group",
                "observed_value": "A",
                "counterfactual_value": "B",
            },
            ranking=("i1", "i3", "i2"),
        ),
        _agg(
            "C2:synthetic_identity_group_A_to_C",
            0.4,
            intervention={
                "attribute": "synthetic_identity_group",
                "observed_value": "A",
                "counterfactual_value": "C",
            },
            ranking=("i3", "i2", "i1"),
        ),
    ]
    pairs = build_fairsynth_identity_pairs(rows)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["rq"] == "SYNTH-ID"
    assert pair["counterfactual_alternatives"] == 2
    assert pair["left_ndcg"] == pytest.approx(0.8)
    assert pair["right_ndcg"] == pytest.approx(0.5)
    assert pair["delta_ndcg"] == pytest.approx(0.3)
    assert pair["real_world_claim_allowed"] is False


def test_identity_reduction_averages_invalid_rate_across_alternatives():
    rows = [
        _agg("C1", 0.8, ranking=("i1", "i2", "i3")),
        _agg(
            "C2:synthetic_identity_group_A_to_B",
            0.0,
            intervention={"attribute": "synthetic_identity_group"},
            ranking=None,
        ),
        _agg(
            "C2:synthetic_identity_group_A_to_C",
            0.5,
            intervention={"attribute": "synthetic_identity_group"},
            ranking=("i2", "i1", "i3"),
        ),
    ]
    pair = build_fairsynth_identity_pairs(rows)[0]
    assert pair["right_invalid_rate"] == pytest.approx(0.5)
    assert pair["invalid_rate_difference"] == pytest.approx(-0.5)


def test_synthetic_personality_pair_is_kept_separate_from_real_rq2():
    c3 = _agg("C3", 0.72, ranking=("i1", "i2", "i3"))
    c3["analysis_roles"] = ["synthetic_personality_sanity"]
    c4 = _agg("C4", 0.52, ranking=("i2", "i1", "i3"))
    c4["analysis_roles"] = ["synthetic_personality_sanity"]
    pairs = build_fairsynth_personality_pairs([c3, c4])
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["rq"] == "SYNTH-PERSONALITY"
    assert pair["contrast"] == "true_synthetic_ocean_vs_shuffled_profile"
    assert pair["delta_ndcg"] == pytest.approx(0.20)
    assert pair["real_world_claim_allowed"] is False
