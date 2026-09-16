import pytest

from faireval.rq4_prompting import (
    build_identity_irrelevance_pairs,
    build_identity_irrelevance_plan,
    summarize_prompting_pairs,
)


def _plan_cell(condition_id: str, *, intervention=None, role=True, prompt_mode="audit"):
    return {
        "schema_version": "faireval-run-plan-v1",
        "cell_id": f"source-{condition_id}",
        "dataset": "movielens_1m",
        "user_id": "u1",
        "condition": {
            "condition_id": condition_id,
            "condition_name": "observed_demographic" if condition_id == "C1" else "counterfactual_demographic",
            "demographics": {"gender": "female"},
            "personality": None,
            "intervention": intervention or {},
        },
        "analysis_roles": ["rq4_mitigation_baseline"] if role else ["other"],
        "confirmatory": True,
        "model_family": "openai",
        "model_id": "gpt-5.6-terra",
        "reasoning_or_thinking_setting": "none",
        "sampling_policy": "explicit_where_supported",
        "output_token_parameter": "max_completion_tokens",
        "template_id": "field_v2_a",
        "prompt_mode": prompt_mode,
        "cue_id": "structured_key_value",
        "candidate_order_seed": 7,
        "k": 10,
        "repetition": 0,
        "temperature": 0.2,
        "top_p": 1.0,
        "max_output_tokens": 512,
    }


def _aggregated(condition_id: str, *, ndcg: float, intervention=None):
    return {
        "dataset": "movielens_1m",
        "user_id": "u1",
        "model_family": "openai",
        "requested_model_id": "gpt-5.6-terra",
        "condition_id": condition_id,
        "condition_intervention": intervention or {},
        "analysis_roles": ["rq4_identity_irrelevance_prompting"],
        "prompt_template_id": "field_v2_a",
        "prompt_mode": "identity_irrelevance",
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


def test_prompt_plan_changes_only_named_intervention_mode_and_keeps_source_link():
    gender_cf = {
        "type": "demographic_counterfactual",
        "attribute": "gender",
        "observed_value": "female",
        "counterfactual_value": "male",
    }
    age_cf = {
        "type": "demographic_counterfactual",
        "attribute": "age_group",
        "observed_value": "25_34",
        "counterfactual_value": "35_44",
    }
    rows = build_identity_irrelevance_plan(
        [
            _plan_cell("C1"),
            _plan_cell("C2:gender_female_to_male", intervention=gender_cf),
            _plan_cell("C2:age_group_25_34_to_35_44", intervention=age_cf),
        ]
    )
    assert len(rows) == 2
    assert all(row["prompt_mode"] == "identity_irrelevance" for row in rows)
    assert all(row["rq4_intervention"] == "identity_irrelevance_prompting" for row in rows)
    assert all(row["source_audit_cell_id"].startswith("source-") for row in rows)
    assert all(row["confirmatory"] is False for row in rows)


def test_prompt_pairs_and_macro_summary_are_user_level():
    gender_cf = {
        "type": "demographic_counterfactual",
        "attribute": "gender",
        "observed_value": "female",
        "counterfactual_value": "male",
    }
    pairs = build_identity_irrelevance_pairs(
        [
            _aggregated("C1", ndcg=0.8),
            _aggregated("C2:gender_female_to_male", ndcg=0.6, intervention=gender_cf),
        ]
    )
    assert len(pairs) == 1
    assert pairs[0]["delta_ndcg"] == pytest.approx(0.2)
    summary = summarize_prompting_pairs(pairs)
    assert summary["macro"]["mean_identity_ndcg"] == pytest.approx(0.7)
    assert summary["macro"]["mean_abs_cug_ndcg"] == pytest.approx(0.2)
    assert summary["confirmatory_p_values"] is False


def test_prompt_plan_rejects_non_audit_source_cells():
    with pytest.raises(ValueError, match="neutral audit"):
        build_identity_irrelevance_plan([_plan_cell("C1", prompt_mode="identity_irrelevance")])
