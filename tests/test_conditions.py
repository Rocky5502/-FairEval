import pytest

from faireval.conditions import (
    build_personality_derangement,
    demographic_counterfactual,
    observed_demographic,
    one_trait_counterfactual,
    preference_only,
    shuffled_personality,
    true_personality,
)
from faireval.schema import Item, PersonalityProfile, UserInstance


def _user(user_id, *, openness, gender="female"):
    return UserInstance(
        dataset="toy",
        user_id=user_id,
        history=[Item("h", "History")],
        candidates=[Item("a", "A"), Item("b", "B")],
        relevant_item_ids=frozenset({"a"}),
        demographics={"gender": gender, "age_group": "25_34"},
        personality=PersonalityProfile(
            openness=openness,
            conscientiousness=0.4,
            extraversion=0.5,
            agreeableness=0.6,
            neuroticism=0.3,
        ),
    )


def test_basic_conditions_keep_domains_separate():
    user = _user("u1", openness=0.2)
    c0 = preference_only()
    c1 = observed_demographic(user)
    c3 = true_personality(user)
    assert c0.demographics is None and c0.personality is None
    assert c1.demographics == user.demographics and c1.personality is None
    assert c3.demographics is None and c3.personality == user.personality


def test_demographic_counterfactual_changes_exactly_requested_field():
    user = _user("u1", openness=0.2)
    cf = demographic_counterfactual(
        user,
        attribute="gender",
        counterfactual_value="male",
        counterfactual_id="gender_swap",
    )
    assert cf.demographics == {"gender": "male", "age_group": "25_34"}
    assert cf.intervention["attribute"] == "gender"
    assert cf.intervention["observed_value"] == "female"
    assert cf.intervention["counterfactual_value"] == "male"


def test_personality_derangement_is_deterministic_and_has_no_self_donors():
    users = [_user(f"u{i}", openness=i / 10) for i in range(1, 6)]
    first = build_personality_derangement(users, seed=2027)
    second = build_personality_derangement(list(reversed(users)), seed=2027)
    assert first == second
    assert set(first) == {u.user_id for u in users}
    assert all(target != donor for target, donor in first.items())


def test_shuffled_and_one_trait_conditions_are_falsifiable_controls():
    target = _user("u1", openness=0.2)
    donor = _user("u2", openness=0.8, gender="male")
    shuffled = shuffled_personality(target, donor_instance=donor)
    assert shuffled.personality == donor.personality
    assert shuffled.demographics is None

    one_trait = one_trait_counterfactual(target, trait="openness", donor_instance=donor)
    assert one_trait.personality is not None
    assert one_trait.personality.openness == pytest.approx(0.8)
    assert one_trait.personality.conscientiousness == pytest.approx(0.4)
    assert one_trait.personality.extraversion == pytest.approx(0.5)
    assert one_trait.intervention["trait"] == "openness"


def test_one_trait_rejects_noop_donor():
    target = _user("u1", openness=0.2)
    donor = _user("u2", openness=0.2)
    with pytest.raises(ValueError, match="identical openness"):
        one_trait_counterfactual(target, trait="openness", donor_instance=donor)
