import pytest

from scripts.render_result_tables import (
    render_fairsynth_table,
    render_rq12_main,
    render_rq34_main,
    render_trait_table,
    render_whitebox_table,
)


def test_rq12_contract_contains_all_eight_model_rows():
    text = render_rq12_main(None)
    for token in (
        "OpenAI",
        "Anthropic",
        "Google",
        "DeepSeek",
        "Qwen API",
        "Llama",
        "Qwen2.5-7B local",
        "Phi-3.5-mini local",
    ):
        assert token in text
    assert "RQ1 MovieLens-1M" in text
    assert "RQ2 REASONER" in text
    assert text.count("\\tbd") == 40


def test_trait_table_uses_attribute_specific_inference():
    rows = []
    for trait, value in [
        ("openness", 0.10),
        ("conscientiousness", 0.05),
        ("extraversion", -0.02),
        ("agreeableness", 0.03),
        ("neuroticism", -0.01),
    ]:
        rows.append(
            {
                "contrast": "true_vs_one_trait_counterfactual",
                "metric": "ndcg",
                "attribute": trait,
                "mean_paired_difference": value,
            }
        )
    text = render_trait_table(rows)
    assert "Openness & 1 & 0.100" in text
    assert "Extraversion & 1 & -0.020" in text


def test_fairsynth_table_keeps_hosted_and_local_strata_separate():
    rows = []
    for rq, contrast in [
        ("SYNTH-ID", "observed_identity_vs_mean_counterfactual_identity"),
        ("SYNTH-PERSONALITY", "true_synthetic_ocean_vs_shuffled_profile"),
    ]:
        for family, effect in [("openai", 0.01), ("qwen25_local", -0.02)]:
            rows.append(
                {
                    "rq": rq,
                    "contrast": contrast,
                    "metric": "ndcg",
                    "model_family": family,
                    "mean_paired_difference": effect,
                }
            )
    text = render_fairsynth_table(rows)
    assert "Identity sanity / hosted & 1" in text
    assert "Identity sanity / local & 1" in text
    assert "Synthetic personality sanity / local & 1" in text


def test_whitebox_table_is_explicitly_exploratory():
    rows = [
        {
            "mean_generated_token_logprob_spearman_ndcg": {"rho": 0.4},
            "generated_token_nll_spearman_ndcg": {"rho": -0.4},
            "generated_token_perplexity_spearman_ndcg": {"rho": -0.3},
            "mean_top1_top2_logit_margin_spearman_ndcg": {"rho": 0.2},
        }
    ]
    text = render_whitebox_table(rows)
    assert "Exploratory local white-box diagnostics" in text
    assert "Mean token log-prob. & 1 & 0.400" in text


def _pair_artifact(*, selected_on_test: bool = False):
    return {
        "schema_version": "faireval-rq4-pair-artifact-v1",
        "selection_used_test_outcomes": selected_on_test,
        "per_model_or_dataset_tuning": False,
        "operating_point": {
            "alpha": 0.5,
            "lambda_instability": 0.2,
            "validation_summary": {"utility_retention": 0.97},
        },
        "test_summary": {
            "n_users": 80,
            "baseline_identity_ndcg_mean": 0.60,
            "baseline_abs_cug_ndcg_on_available": 0.08,
            "pair_identity_ndcg_mean_system": 0.58,
            "utility_retention": 0.967,
            "pair_abs_cug_ndcg_on_available": 0.03,
        },
    }


def test_rq34_refuses_test_selected_pair_artifact():
    with pytest.raises(ValueError, match="test outcomes"):
        render_rq34_main(None, _pair_artifact(selected_on_test=True))


def test_rq34_populates_identity_prompting_from_dedicated_summary():
    prompting = {
        "schema_version": "faireval-rq4-prompting-summary-v1",
        "intervention": "identity_irrelevance_prompting",
        "confirmatory_p_values": False,
        "macro": {
            "n_strata": 12,
            "mean_identity_ndcg": 0.59,
            "mean_abs_cug_ndcg": 0.05,
            "mean_invalid_rate": 0.01,
        },
    }
    text = render_rq34_main(None, _pair_artifact(), prompting)
    assert "Identity-irrelevance prompting & 12 strata & nDCG=0.590" in text
    assert "$|\\CUG|$=0.050" in text
    assert "missing prompting artifact" not in text


def test_rq34_rejects_prompting_summary_with_confirmatory_claim():
    prompting = {
        "schema_version": "faireval-rq4-prompting-summary-v1",
        "intervention": "identity_irrelevance_prompting",
        "confirmatory_p_values": True,
        "macro": {},
    }
    with pytest.raises(ValueError, match="confirmatory_p_values=false"):
        render_rq34_main(None, _pair_artifact(), prompting)
