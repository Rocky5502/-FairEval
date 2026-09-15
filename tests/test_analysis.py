import pytest

from faireval.analysis import aggregate_repetitions, build_rq1_confirmatory_pairs, build_rq2_pairs


def _scored(
    condition_id: str,
    repetition: int,
    ndcg: float,
    *,
    invalid: bool = False,
    confirmatory: bool = True,
    intervention=None,
):
    valid_only = None if invalid else ndcg
    return {
        "schema_version": "faireval-scored-run-v1",
        "plan_sha256": "plan",
        "planned_cell_id": f"cell-{condition_id}-{repetition}",
        "dataset": "movielens_1m",
        "user_id": "u1",
        "model_family": "openai",
        "requested_model_id": "gpt-5.6-terra",
        "condition_id": condition_id,
        "condition_name": {
            "C0": "preference_only",
            "C1": "observed_demographic",
            "C3": "true_personality",
            "C4": "shuffled_personality",
        }.get(condition_id, "counterfactual_demographic"),
        "condition_intervention": intervention or {},
        "analysis_roles": ["test"],
        "confirmatory": confirmatory,
        "prompt_template_id": "field_v2_a",
        "prompt_mode": "audit",
        "cue_id": "structured_key_value",
        "candidate_order_seed": None,
        "k": 10,
        "repetition": repetition,
        "final_valid": not invalid,
        "invalid_output": invalid,
        "ndcg": ndcg,
        "recall": ndcg,
        "mrr": ndcg,
        "valid_only_ndcg": valid_only,
        "valid_only_recall": valid_only,
        "valid_only_mrr": valid_only,
        "code_commit_sha": "abc",
    }


def test_repetition_aggregation_keeps_invalid_zero_in_primary_but_not_valid_only():
    rows = [
        _scored("C1", 0, 0.8),
        _scored("C1", 1, 0.0, invalid=True),
    ]
    aggregated = aggregate_repetitions(rows)
    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["ndcg"] == pytest.approx(0.4)
    assert row["valid_only_ndcg"] == pytest.approx(0.8)
    assert row["invalid_rate"] == pytest.approx(0.5)
    assert row["repetitions"] == 2


def test_rq1_pairs_observed_with_confirmatory_counterfactual_only():
    intervention = {
        "type": "demographic_counterfactual",
        "attribute": "gender",
        "observed_value": "female",
        "counterfactual_value": "male",
    }
    rows = aggregate_repetitions(
        [
            _scored("C1", 0, 0.8),
            _scored("C2:gender_female_to_male", 0, 0.5, intervention=intervention),
            _scored(
                "C2:age_group_25_34_to_35_44",
                0,
                0.1,
                confirmatory=False,
                intervention={
                    "type": "demographic_counterfactual",
                    "attribute": "age_group",
                    "observed_value": "25_34",
                    "counterfactual_value": "35_44",
                },
            ),
        ]
    )
    pairs = build_rq1_confirmatory_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["attribute"] == "gender"
    assert pairs[0]["delta_ndcg"] == pytest.approx(0.3)
    assert pairs[0]["contrast"] == "observed_vs_demographic_counterfactual"


def test_rq2_builds_true_vs_shuffled_and_true_vs_preference_contrasts():
    def row(condition_id, score):
        value = _scored(condition_id, 0, score)
        value["dataset"] = "personality2018"
        return value

    aggregated = aggregate_repetitions(
        [
            row("C0", 0.55),
            row("C3", 0.75),
            row("C4", 0.60),
        ]
    )
    pairs = build_rq2_pairs(aggregated)
    by_contrast = {pair["contrast"]: pair for pair in pairs}
    assert by_contrast["true_vs_shuffled_personality"]["delta_ndcg"] == pytest.approx(0.15)
    assert by_contrast["true_personality_vs_preference_only"]["delta_ndcg"] == pytest.approx(0.20)
