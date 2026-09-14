import json

from faireval.evaluator import validate_ranking_output
from faireval.prompts import build_ranking_prompt
from faireval.schema import Item, PersonalityProfile, PromptCondition, UserInstance


def _instance() -> UserInstance:
    candidates = [
        Item("i1", "One"),
        Item("i2", "Two"),
        Item("i3", "Three"),
    ]
    return UserInstance(
        dataset="toy",
        user_id="u1",
        history=[Item("h1", "History")],
        candidates=candidates,
        relevant_item_ids=frozenset({"i2"}),
    )


def test_prompt_has_fixed_context_fields():
    instance = _instance()
    neutral = PromptCondition("C0", "neutral")
    personality = PromptCondition(
        "C3",
        "true_personality",
        personality=PersonalityProfile(0.8, 0.4, 0.2, 0.6, 0.5),
    )
    p0 = build_ranking_prompt(instance, neutral, k=2)
    p3 = build_ranking_prompt(instance, personality, k=2)
    for field in ["demographic_context", "personality_ocean", "candidate_items", "preference_history"]:
        assert field in p0
        assert field in p3
    assert '"personality_ocean":"unspecified"' in p0
    assert '"openness":0.8' in p3


def test_validator_accepts_only_exact_candidate_ranking():
    instance = _instance()
    good = json.dumps({"ranked_item_ids": ["i2", "i1"]})
    result = validate_ranking_output(good, instance, k=2)
    assert result.valid
    assert result.ranking is not None
    assert result.ranking.ranked_item_ids == ("i2", "i1")


def test_validator_keeps_failure_reason():
    instance = _instance()
    bad = json.dumps({"ranked_item_ids": ["i2", "invented"]})
    result = validate_ranking_output(bad, instance, k=2)
    assert not result.valid
    assert "out_of_candidate_item" in result.errors
