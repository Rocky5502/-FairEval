import json

from faireval.evaluator import validate_ranking_output
from faireval.prompts import (
    AUDIT_SYSTEM_INSTRUCTION,
    IDENTITY_IRRELEVANCE_SYSTEM_INSTRUCTION,
    PROMPT_TEMPLATES,
    build_ranking_prompt,
    changed_payload_fields,
    parse_prompt_payload,
)
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


def test_audit_prompt_contains_no_fairness_coaching():
    lower = AUDIT_SYSTEM_INSTRUCTION.lower()
    for forbidden in ["fair", "stereotype", "protected", "demographic identity"]:
        assert forbidden not in lower


def test_fairness_language_is_isolated_to_named_mitigation():
    assert "demographic identity" in IDENTITY_IRRELEVANCE_SYSTEM_INSTRUCTION.lower()
    audit = build_ranking_prompt(_instance(), PromptCondition("C0", "neutral"), k=2)
    mitigated = build_ranking_prompt(
        _instance(),
        PromptCondition("C0", "neutral"),
        k=2,
        prompt_mode="identity_irrelevance",
    )
    assert audit != mitigated
    assert "stereotype" not in audit.lower()
    assert "stereotype" in mitigated.lower()


def test_pure_demographic_counterfactual_changes_one_payload_field_only():
    instance = _instance()
    left = build_ranking_prompt(
        instance,
        PromptCondition("C2a", "cf_gender_a", demographics={"gender": "A"}),
        k=2,
    )
    right = build_ranking_prompt(
        instance,
        PromptCondition("C2b", "cf_gender_b", demographics={"gender": "B"}),
        k=2,
    )
    assert changed_payload_fields(left, right) == ("demographic_context",)


def test_pure_personality_counterfactual_changes_one_payload_field_only():
    instance = _instance()
    p1 = PersonalityProfile(0.8, 0.4, 0.2, 0.6, 0.5)
    p2 = PersonalityProfile(0.2, 0.4, 0.2, 0.6, 0.5)
    left = build_ranking_prompt(instance, PromptCondition("C5a", "trait_a", personality=p1), k=2)
    right = build_ranking_prompt(instance, PromptCondition("C5b", "trait_b", personality=p2), k=2)
    assert changed_payload_fields(left, right) == ("personality_ocean",)


def test_all_templates_keep_identical_structural_fields_and_candidate_order():
    instance = _instance()
    condition = PromptCondition("C0", "neutral")
    prompts = {
        template_id: build_ranking_prompt(instance, condition, k=2, template_id=template_id)
        for template_id in PROMPT_TEMPLATES
    }
    payloads = {name: parse_prompt_payload(prompt) for name, prompt in prompts.items()}
    reference = next(iter(payloads.values()))
    for payload in payloads.values():
        assert tuple(payload.keys()) == tuple(reference.keys())
        assert payload["candidate_items"] == reference["candidate_items"]
        assert payload["preference_history"] == reference["preference_history"]
        assert payload["output_contract"] == reference["output_contract"]


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
