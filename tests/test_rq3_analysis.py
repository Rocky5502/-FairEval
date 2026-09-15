import pytest

from faireval.rq3_analysis import build_factor_rows, summarize_factor_variation


def _scored(*, cell: str, condition: str, repetition: int, ndcg: float, invalid: bool = False):
    return {
        "planned_cell_id": cell,
        "dataset": "movielens_1m",
        "user_id": "u1",
        "model_family": "openai",
        "requested_model_id": "gpt-5.6-terra",
        "condition_id": condition,
        "repetition": repetition,
        "invalid_output": invalid,
        "ndcg": ndcg,
        "recall": ndcg,
        "mrr": ndcg,
        "code_commit_sha": "abc123",
    }


def test_prompt_factor_reuses_core_rep0_and_summarizes_within_user_variation():
    core = [_scored(cell="core", condition="C1", repetition=0, ndcg=0.8)]
    extra = [
        _scored(cell="b", condition="C1", repetition=0, ndcg=0.7),
        _scored(cell="c", condition="C1", repetition=0, ndcg=0.5),
    ]
    plan = {
        "b": {"robustness_factor": "prompt", "robustness_level": "field_v2_b"},
        "c": {"robustness_factor": "prompt", "robustness_level": "field_v2_c"},
    }
    rows = build_factor_rows(core_scored=core, rq3_scored=extra, rq3_plan_by_cell=plan)
    ndcg = [row for row in rows if row["metric"] == "ndcg"]
    assert {row["level"] for row in ndcg} == {"field_v2_a", "field_v2_b", "field_v2_c"}
    summary = summarize_factor_variation(rows)
    result = next(row for row in summary if row["metric"] == "ndcg")
    assert result["n_users"] == 1
    assert result["mean_within_user_range"] == pytest.approx(0.3)
    assert result["mean_within_user_sd"] > 0


def test_stochasticity_uses_repeated_extra_cells_without_core_baseline():
    extra = [
        _scored(cell=f"s{i}", condition="C1", repetition=i, ndcg=value)
        for i, value in enumerate([0.5, 0.7, 0.6])
    ]
    plan = {
        f"s{i}": {"robustness_factor": "stochasticity", "robustness_level": "high_stochasticity"}
        for i in range(3)
    }
    rows = build_factor_rows(core_scored=[], rq3_scored=extra, rq3_plan_by_cell=plan)
    summary = summarize_factor_variation(rows)
    result = next(row for row in summary if row["metric"] == "ndcg")
    assert result["n_users"] == 1
    assert result["mean_within_user_range"] == pytest.approx(0.2)


def test_factor_analysis_rejects_cross_commit_comparison():
    core = [_scored(cell="core", condition="C1", repetition=0, ndcg=0.8)]
    extra = [_scored(cell="b", condition="C1", repetition=0, ndcg=0.7)]
    extra[0]["code_commit_sha"] = "different"
    plan = {"b": {"robustness_factor": "prompt", "robustness_level": "field_v2_b"}}
    with pytest.raises(ValueError, match="crosses code commits"):
        build_factor_rows(core_scored=core, rq3_scored=extra, rq3_plan_by_cell=plan)
