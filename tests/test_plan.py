from pathlib import Path

import yaml

from faireval.plan import expand_core_run_cells, load_model_panel, plan_core_conditions
from faireval.schema import Item, PersonalityProfile, UserInstance


def _base_instance(
    dataset: str,
    user_id: str,
    *,
    gender: str | None = None,
    age_group: str | None = None,
    personality: PersonalityProfile | None = None,
) -> UserInstance:
    demographics = {}
    if gender is not None:
        demographics["gender"] = gender
    if age_group is not None:
        demographics["age_group"] = age_group
    return UserInstance(
        dataset=dataset,
        user_id=user_id,
        history=[Item("h", "History")],
        candidates=[Item("a", "A"), Item("b", "B"), Item("c", "C")],
        relevant_item_ids=frozenset({"a"}),
        demographics=demographics,
        personality=personality,
    )


def _profile(index: int) -> PersonalityProfile:
    # Every trait varies, ensuring each one-trait donor plan is testable.
    base = index / 10.0
    return PersonalityProfile(
        openness=base,
        conscientiousness=(index + 1) / 10.0,
        extraversion=(index + 2) / 10.0,
        agreeableness=(index + 3) / 10.0,
        neuroticism=(index + 4) / 10.0,
    )


def test_personality_plan_has_true_shuffled_and_frozen_trait_controls():
    users = [
        _base_instance("personality2018", f"u{i}", personality=_profile(i))
        for i in range(1, 6)
    ]
    rows = plan_core_conditions(
        users,
        counterfactual_config={},
        seed=2027,
        one_trait_subset_users=2,
    )
    by_user = {}
    for row in rows:
        by_user.setdefault(row.user_id, []).append(row)

    for user_id in by_user:
        condition_ids = {row.condition.condition_id for row in by_user[user_id]}
        assert {"C0", "C3", "C4"}.issubset(condition_ids)

    trait_rows = [row for row in rows if row.condition.condition_id.startswith("C5:")]
    assert len(trait_rows) == 2 * 5
    for row in trait_rows:
        assert row.confirmatory is False
        assert row.condition.intervention["other_traits_held_fixed"] is True


def test_demographic_plan_uses_gender_confirmatory_and_age_robustness():
    users = [
        _base_instance(
            "movielens_1m",
            "u1",
            gender="female",
            age_group="25_34",
        ),
        _base_instance(
            "movielens_1m",
            "u2",
            gender="male",
            age_group="35_44",
        ),
    ]
    cfg = {
        "movielens_1m": {
            "confirmatory": {"gender": {"values": ["female", "male"]}},
            "robustness": {
                "age_group": {
                    "values": ["18_24", "25_34", "35_44", "45_49"],
                }
            },
        }
    }
    rows = plan_core_conditions(
        users,
        counterfactual_config=cfg,
        seed=2027,
        demographic_robustness_subset_users=1,
    )

    gender_cf = [
        row
        for row in rows
        if row.condition.intervention.get("attribute") == "gender"
    ]
    assert len(gender_cf) == 2
    assert all(row.confirmatory for row in gender_cf)
    assert all("rq1_demographic" in row.analysis_roles for row in gender_cf)

    age_cf = [
        row
        for row in rows
        if row.condition.intervention.get("attribute") == "age_group"
    ]
    # Exactly one user is in the frozen robustness subset and receives all three
    # non-observed age alternatives from the four-category test taxonomy.
    assert len(age_cf) == 3
    assert all(not row.confirmatory for row in age_cf)
    assert all(row.analysis_roles == ("rq1_demographic_robustness",) for row in age_cf)


def test_model_panel_and_run_cells_are_unique(tmp_path: Path):
    model_rows = [
        {
            "family": family,
            "model_id": f"{family}-model",
            "enabled": True,
            "reasoning_or_thinking_setting": (
                "low" if family == "google" else "not_applicable" if family == "meta" else "disabled"
            ),
            "sampling_policy": "test_policy",
        }
        for family in ("openai", "anthropic", "google", "deepseek", "qwen", "meta")
    ]
    model_path = tmp_path / "models.yaml"
    model_path.write_text(yaml.safe_dump({"models": model_rows}), encoding="utf-8")
    panel = load_model_panel(model_path)
    assert len(panel) == 6
    assert all("reasoning_or_thinking_setting" in model for model in panel)
    assert all("sampling_policy" in model for model in panel)

    user = _base_instance(
        "movielens_1m",
        "u1",
        gender="female",
        age_group="25_34",
    )
    cfg = {
        "movielens_1m": {
            "confirmatory": {"gender": {"values": ["female", "male"]}},
            "robustness": {},
        }
    }
    conditions = plan_core_conditions(
        [user],
        counterfactual_config=cfg,
        seed=2027,
        demographic_robustness_subset_users=0,
    )
    cells = expand_core_run_cells(conditions, model_panel=panel, repetitions=3)
    assert len(cells) == len(conditions) * 6 * 3
    assert len({cell["cell_id"] for cell in cells}) == len(cells)
    assert all(cell["prompt_mode"] == "audit" for cell in cells)
    assert all(cell["sampling_policy"] == "test_policy" for cell in cells)
    assert all(cell["reasoning_or_thinking_setting"] for cell in cells)
