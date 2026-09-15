from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .schema import OCEAN_KEYS, PersonalityProfile, PromptCondition, UserInstance


def _stable_int(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _validate_single_dataset(instances: Sequence[UserInstance]) -> None:
    datasets = {instance.dataset for instance in instances}
    if len(datasets) != 1:
        raise ValueError(
            "condition controls must be planned within one dataset; "
            f"received datasets={sorted(datasets)!r}"
        )


def preference_only() -> PromptCondition:
    return PromptCondition(
        condition_id="C0",
        condition_name="preference_only",
        demographics=None,
        personality=None,
        intervention={"type": "none"},
    )


def observed_demographic(instance: UserInstance) -> PromptCondition:
    if not instance.demographics:
        raise ValueError(f"{instance.dataset}/{instance.user_id} has no observed demographics")
    return PromptCondition(
        condition_id="C1",
        condition_name="observed_demographic",
        demographics=dict(instance.demographics),
        personality=None,
        intervention={"type": "observed_context"},
    )


def demographic_counterfactual(
    instance: UserInstance,
    *,
    attribute: str,
    counterfactual_value: Any,
    counterfactual_id: str | None = None,
) -> PromptCondition:
    """Change exactly one observed demographic field in prompt context.

    This is a benchmark intervention, not a claim that a social identity is
    literally mutable. The caller must choose values from a pre-registered
    dataset-specific counterfactual plan; this function intentionally does not
    invent or infer alternatives.
    """
    if attribute not in instance.demographics:
        raise ValueError(f"attribute {attribute!r} is not observed for this user")
    observed_value = instance.demographics[attribute]
    if counterfactual_value == observed_value:
        raise ValueError("counterfactual value must differ from the observed value")

    changed = dict(instance.demographics)
    changed[attribute] = counterfactual_value
    suffix = counterfactual_id or f"{attribute}_cf"
    return PromptCondition(
        condition_id=f"C2:{suffix}",
        condition_name="counterfactual_demographic",
        demographics=changed,
        personality=None,
        intervention={
            "type": "demographic_counterfactual",
            "attribute": attribute,
            "observed_value": observed_value,
            "counterfactual_value": counterfactual_value,
        },
    )


def all_categorical_demographic_counterfactuals(
    instance: UserInstance,
    *,
    attribute: str,
    allowed_values: Sequence[Any],
) -> tuple[PromptCondition, ...]:
    """Generate every pre-registered non-observed category for one attribute.

    Generating the full declared alternative set prevents selecting a comparison
    after seeing which counterfactual produces the largest disparity. The caller
    owns the dataset-native category definition; no category is inferred here.
    """
    if attribute not in instance.demographics:
        raise ValueError(f"attribute {attribute!r} is not observed for this user")
    values = tuple(allowed_values)
    if len(values) < 2 or len(set(values)) != len(values):
        raise ValueError("allowed_values must contain at least two unique values")
    observed = instance.demographics[attribute]
    if observed not in values:
        raise ValueError(
            f"observed value {observed!r} is not present in allowed_values={values!r}"
        )
    return tuple(
        demographic_counterfactual(
            instance,
            attribute=attribute,
            counterfactual_value=value,
            counterfactual_id=f"{attribute}_{observed}_to_{value}",
        )
        for value in values
        if value != observed
    )


def true_personality(instance: UserInstance) -> PromptCondition:
    if instance.personality is None:
        raise ValueError(f"{instance.dataset}/{instance.user_id} has no measured personality")
    # Validate normalized scores before they enter any prompt.
    instance.personality.as_dict()
    return PromptCondition(
        condition_id="C3",
        condition_name="true_personality",
        demographics=None,
        personality=instance.personality,
        intervention={"type": "measured_personality"},
    )


def build_personality_derangement(
    instances: Sequence[UserInstance],
    *,
    seed: int,
) -> dict[str, str]:
    """Return a deterministic user->donor mapping with no self-donors.

    Users without measured personality are rejected rather than silently removed.
    A hash-sorted cyclic shift is used so the mapping is deterministic across
    Python versions and independent of input list order. Because the mapping is a
    permutation, the cohort's full multivariate personality distribution and
    real cross-trait covariance are preserved exactly.
    """
    if len(instances) < 2:
        raise ValueError("at least two users are required for a personality derangement")
    _validate_single_dataset(instances)
    ids = [str(instance.user_id) for instance in instances]
    if len(set(ids)) != len(ids):
        raise ValueError("user IDs must be unique within a shuffle pool")
    for instance in instances:
        if instance.personality is None:
            raise ValueError(f"user {instance.user_id} lacks measured personality")
        instance.personality.as_dict()

    ordered = sorted(ids, key=lambda user_id: (_stable_int(seed, "personality", user_id), user_id))
    # Pick a non-zero deterministic rotation. This is a true derangement for a
    # cyclic list: no position maps to itself for shift in [1, n-1].
    shift = 1 + (_stable_int(seed, "rotation", len(ordered)) % (len(ordered) - 1))
    return {
        user_id: ordered[(idx + shift) % len(ordered)]
        for idx, user_id in enumerate(ordered)
    }


def shuffled_personality(
    instance: UserInstance,
    *,
    donor_instance: UserInstance,
) -> PromptCondition:
    if instance.dataset != donor_instance.dataset:
        raise ValueError("personality donor must come from the same dataset")
    if instance.personality is None:
        raise ValueError("target user lacks measured personality")
    if donor_instance.personality is None:
        raise ValueError("donor user lacks measured personality")
    if str(instance.user_id) == str(donor_instance.user_id):
        raise ValueError("shuffled personality donor must be a different user")
    donor_instance.personality.as_dict()
    return PromptCondition(
        condition_id="C4",
        condition_name="shuffled_personality",
        demographics=None,
        personality=donor_instance.personality,
        intervention={
            "type": "personality_shuffle",
            "donor_user_id": str(donor_instance.user_id),
            "preserves_cohort_profile_distribution": True,
        },
    )


def build_one_trait_donor_map(
    instances: Sequence[UserInstance],
    *,
    trait: str,
    seed: int,
) -> dict[str, str]:
    """Freeze a deterministic distant-quantile donor for one OCEAN trait.

    Users are sorted by the measured trait with a seeded stable tie-break. The
    first donor candidate is a half-cohort cyclic shift, so it comes from roughly
    the opposite half of the empirical distribution. If tied values make that a
    no-op, the search proceeds deterministically around the cycle until a user
    with a different observed trait value is found. No fabricated 0/1 extremes
    and no result-aware donor choice are permitted.
    """
    if trait not in OCEAN_KEYS:
        raise ValueError(f"trait must be one of {OCEAN_KEYS}, got {trait!r}")
    if len(instances) < 3:
        raise ValueError("at least three users are required for one-trait donor planning")
    _validate_single_dataset(instances)
    ids = [str(instance.user_id) for instance in instances]
    if len(set(ids)) != len(ids):
        raise ValueError("user IDs must be unique within a donor pool")
    for instance in instances:
        if instance.personality is None:
            raise ValueError(f"user {instance.user_id} lacks measured personality")
        instance.personality.as_dict()

    ordered = sorted(
        instances,
        key=lambda instance: (
            float(getattr(instance.personality, trait)),  # type: ignore[arg-type]
            _stable_int(seed, "one_trait", trait, instance.user_id),
            str(instance.user_id),
        ),
    )
    values = [float(getattr(instance.personality, trait)) for instance in ordered]  # type: ignore[arg-type]
    if len(set(values)) < 2:
        raise ValueError(f"all users have identical {trait}; a trait intervention is impossible")

    n = len(ordered)
    half_shift = (n + 1) // 2
    result: dict[str, str] = {}
    for index, target in enumerate(ordered):
        target_value = float(getattr(target.personality, trait))  # type: ignore[arg-type]
        donor = None
        # Start at the opposite half, then move deterministically through all
        # non-self candidates. This handles ties without post-hoc selection.
        for extra in range(n - 1):
            candidate = ordered[(index + half_shift + extra) % n]
            if str(candidate.user_id) == str(target.user_id):
                continue
            candidate_value = float(getattr(candidate.personality, trait))  # type: ignore[arg-type]
            if candidate_value != target_value:
                donor = candidate
                break
        if donor is None:  # defensive; global variation check above should prevent this
            raise ValueError(f"no non-identical {trait} donor for user {target.user_id}")
        result[str(target.user_id)] = str(donor.user_id)
    return result


def one_trait_counterfactual(
    instance: UserInstance,
    *,
    trait: str,
    donor_instance: UserInstance,
) -> PromptCondition:
    """Replace one measured Big Five dimension with a frozen donor user's value."""
    if trait not in OCEAN_KEYS:
        raise ValueError(f"trait must be one of {OCEAN_KEYS}, got {trait!r}")
    if instance.dataset != donor_instance.dataset:
        raise ValueError("one-trait donor must come from the same dataset")
    if instance.personality is None or donor_instance.personality is None:
        raise ValueError("target and donor must both have measured personality")
    if str(instance.user_id) == str(donor_instance.user_id):
        raise ValueError("one-trait donor must be a different user")

    target_values = instance.personality.as_dict()
    donor_values = donor_instance.personality.as_dict()
    original = target_values[trait]
    replacement = donor_values[trait]
    if replacement == original:
        # Equal numeric values are possible; returning an unchanged profile would
        # falsely label a no-op as an intervention.
        raise ValueError(f"donor has identical {trait} value; use the frozen donor planner")

    profile = replace(instance.personality, **{trait: replacement})
    profile.as_dict()
    return PromptCondition(
        condition_id=f"C5:{trait}",
        condition_name="one_trait_counterfactual",
        demographics=None,
        personality=profile,
        intervention={
            "type": "one_trait_personality_counterfactual",
            "trait": trait,
            "observed_value": original,
            "counterfactual_value": replacement,
            "donor_user_id": str(donor_instance.user_id),
            "other_traits_held_fixed": True,
        },
    )


def intersectional_condition(
    instance: UserInstance,
    *,
    personality: PersonalityProfile | None = None,
    demographics: Mapping[str, Any] | None = None,
    intervention: Mapping[str, Any] | None = None,
) -> PromptCondition:
    """Construct C6 only from explicitly supplied/observed context.

    C6 is deliberately not auto-generated because dataset support and cell sizes
    must be checked before defining an intersectional contrast.
    """
    demo = dict(instance.demographics if demographics is None else demographics)
    profile = instance.personality if personality is None else personality
    if not demo or profile is None:
        raise ValueError("C6 requires both demographic context and measured personality")
    profile.as_dict()
    return PromptCondition(
        condition_id="C6",
        condition_name="intersectional",
        demographics=demo,
        personality=profile,
        intervention=dict(intervention or {"type": "intersectional_context"}),
    )
