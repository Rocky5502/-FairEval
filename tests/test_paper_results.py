from scripts.render_paper_results import render


def _row(rq: str, contrast: str, dataset: str, family: str, metric: str, effect: float):
    return {
        "rq": rq,
        "contrast": contrast,
        "dataset": dataset,
        "model_family": family,
        "metric": metric,
        "mean_paired_difference": effect,
        "bootstrap_ci_low": effect - 0.05,
        "bootstrap_ci_high": effect + 0.05,
        "holm_adjusted_p": 0.02,
        "matched_rank_biserial": 0.4,
    }


def test_render_produces_two_panel_dataset_level_table_without_manual_tbd():
    rows = [
        _row("RQ1", "observed_vs_demographic_counterfactual", "movielens_1m", "openai", "ndcg", 0.1),
        _row("RQ1", "observed_vs_demographic_counterfactual", "movielens_1m", "openai", "recall", 0.08),
        _row("RQ1", "observed_vs_demographic_counterfactual", "movielens_1m", "openai", "invalid_rate", -0.02),
        _row("RQ2", "true_vs_shuffled_personality", "personality2018", "anthropic", "ndcg", 0.12),
        _row("RQ2", "true_vs_shuffled_personality", "personality2018", "anthropic", "recall", 0.05),
        _row("RQ2", "true_vs_shuffled_personality", "personality2018", "anthropic", "invalid_rate", 0.0),
    ]
    text = render(rows)
    assert "Panel A: RQ1" in text
    assert "Panel B: RQ2" in text
    assert "movielens\\_1m" in text
    assert "personality2018" in text
    assert "OpenAI" in text
    assert "Anthropic" in text
    assert "\\tbd" not in text


def test_render_requires_supported_rq_rows():
    try:
        render([_row("RQ9", "x", "toy", "openai", "ndcg", 0.1)])
    except ValueError as exc:
        assert "neither supported RQ1 nor RQ2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsupported-only input should fail")
