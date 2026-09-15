import pytest

from faireval.whitebox_analysis import summarize_whitebox_rows


def _row(index: int, *, valid: bool, ndcg: float, logprob: float, margin: float):
    return {
        "dataset": "fairsynth360",
        "model_family": "qwen25_local",
        "requested_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "final_valid": valid,
        "invalid_output": not valid,
        "ndcg": ndcg,
        "recall": ndcg,
        "mrr": ndcg,
        "mean_generated_token_logprob": logprob,
        "generated_token_nll": -logprob,
        "generated_token_perplexity": 2.0 + index,
        "mean_top1_top2_logit_margin": margin,
    }


def test_whitebox_summary_is_explicitly_exploratory_and_uncalibrated():
    rows = [
        _row(0, valid=True, ndcg=0.2, logprob=-2.0, margin=0.3),
        _row(1, valid=True, ndcg=0.5, logprob=-1.5, margin=0.7),
        _row(2, valid=True, ndcg=0.9, logprob=-0.5, margin=1.4),
        _row(3, valid=False, ndcg=0.0, logprob=-3.0, margin=0.1),
    ]
    summary = summarize_whitebox_rows(rows)
    assert len(summary) == 1
    result = summary[0]
    assert result["analysis_scope"] == "exploratory_local_whitebox"
    assert result["calibrated_uncertainty"] is False
    assert result["n_runs"] == 4
    assert result["n_valid"] == 3
    assert result["invalid_rate"] == pytest.approx(0.25)
    assert result["mean_generated_token_logprob_valid_mean"] == pytest.approx(
        (-2.0 - 1.5 - 0.5) / 3
    )
    assert result["mean_generated_token_logprob_invalid_mean"] == pytest.approx(-3.0)
    assert result["mean_generated_token_logprob_spearman_ndcg"]["rho"] == pytest.approx(1.0)


def test_whitebox_spearman_is_none_when_not_identifiable():
    rows = [
        _row(0, valid=True, ndcg=0.5, logprob=-1.0, margin=0.5),
        _row(1, valid=True, ndcg=0.5, logprob=-1.0, margin=0.5),
    ]
    result = summarize_whitebox_rows(rows)[0]
    stat = result["mean_generated_token_logprob_spearman_ndcg"]
    assert stat["n"] == 2
    assert stat["rho"] is None
    assert stat["p_value_descriptive"] is None


def test_whitebox_summaries_are_stratified_by_model_and_dataset():
    rows = [
        _row(0, valid=True, ndcg=0.4, logprob=-1.2, margin=0.4),
        _row(1, valid=False, ndcg=0.0, logprob=-2.2, margin=0.1),
    ]
    other = dict(rows[0])
    other["model_family"] = "phi35_local"
    other["requested_model_id"] = "microsoft/Phi-3.5-mini-instruct"
    summaries = summarize_whitebox_rows(rows + [other])
    assert {(row["dataset"], row["model_family"]) for row in summaries} == {
        ("fairsynth360", "qwen25_local"),
        ("fairsynth360", "phi35_local"),
    }
