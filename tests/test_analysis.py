import pytest

from faireval.analysis import (
    aggregate_repetitions,
    build_rq1_confirmatory_pairs,
    build_rq2_pairs,
    validate_ranking_against_frozen_instance,
)
from faireval.schema import Item, UserInstance


def _scored(
    condition_id: str,
    repetition: int,
    ndcg: float,
    *,
    invalid: bool = False,
    confirmatory: bool = True,
    intervention=None,
    ranking=None,
):
    valid_only = None if invalid else ndcg
    if ranking is None:
        ranking = ["a", "b", "c"]
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
        "ranking": [] if invalid else list(ranking),
        "ndcg": ndcg,
        "recall": ndcg,
        "mrr": ndcg,
        "valid_only_ndcg": valid_only,
        "valid_only_recall": valid_only,
        "valid_only_mrr": valid_only,
        "code_commit_sha": "abc",
    }


def _frozen_instance() -> UserInstance:
    return UserInstance(
        dataset="movielens_1m",
        user_id="u1",
        history=[Item("h", "History")],
        candidates=[Item("a", "A"), Item("b", "B"), Item("c", "C")],
        relevant_item_ids=frozenset({"a"}),
    )


def test_analysis_revalidates_frozen_candidate_membership():
    instance = _frozen_instance()
    assert validate_ranking_against_frozen_instance(["a", "b"], instance, k=2) == (
        "a",
        "b",
    )
    with pytest.raises(ValueError, match="outside frozen candidates"):
        validate_ranking_against_frozen_instance(["a", "x"], instance, k=2)
    with pytest.raises(ValueError, match="duplicate"):
        validate_ranking_against_frozen_instance(["a", "a"], instance, k=2)
    with pytest.raises(ValueError, match="ranking length"):
        validate_ranking_against_frozen_instance(["a"], instance, k=2)


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
    assert row["ranking_by_repetition"][0]["ranking"] == ["a", "b", "c"]
    assert row["ranking_by_repetition"][1]["valid"] is False


def test_rq1_pairs_observed_with_confirmatory_counterfactual_only():
    intervention = {
        "type": "demographic_counterfactual",
        "attribute": "gender",
        "observed_value": "female",
        "counterfactual_value": "male",
    }
    rows = aggregate_repetitions(
        [
            _scored("C1", 0, 0.8, ranking=["a", "b", "c"]),
            _scored(
                "C2:gender_female_to_male",
                0,
                0.5,
                intervention=intervention,
                ranking=["b", "a", "c"],
            ),
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
    assert pairs[0]["behavioral_valid_pair_count"] == 1
    assert 0.0 < pairs[0]["mean_rbo_at_k"] < 1.0
    assert pairs[0]["mean_one_minus_rbo"] == pytest.approx(1.0 - pairs[0]["mean_rbo_at_k"])


def test_behavioral_diagnostics_exclude_invalid_matched_repetition():
    intervention = {
        "type": "demographic_counterfactual",
        "attribute": "gender",
        "observed_value": "female",
        "counterfactual_value": "male",
    }
    rows = aggregate_repetitions(
        [
            _scored("C1", 0, 0.8, ranking=["a", "b", "c"]),
            _scored("C1", 1, 0.7, ranking=["a", "c", "b"]),
            _scored(
                "C2:gender_female_to_male",
                0,
                0.5,
                intervention=intervention,
                ranking=["b", "a", "c"],
            ),
            _scored(
                "C2:gender_female_to_male",
                1,
                0.0,
                invalid=True,
                intervention=intervention,
            ),
        ]
    )
    pair = build_rq1_confirmatory_pairs(rows)[0]
    assert pair["behavioral_valid_pair_count"] == 1
    assert pair["right_invalid_rate"] == pytest.approx(0.5)


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
