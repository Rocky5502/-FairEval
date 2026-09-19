from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .base import DatasetAdapter, DatasetCard
from ..schema import Item, PersonalityProfile, UserInstance


DATASET_SEED = 1729
N_USERS = 360
N_ITEMS = 120
FEATURE_NAMES = ("exploration", "structure", "social", "calm", "novelty")
IDENTITY_GROUPS = ("A", "B", "C")


def _stable_int(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class _SyntheticUser:
    user_id: str
    identity_group: str
    personality: PersonalityProfile
    preference_vector: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class _SyntheticItem:
    item_id: str
    title: str
    features: tuple[float, float, float, float, float]


def _identity_assignments() -> list[str]:
    values = ["A"] * 120 + ["B"] * 120 + ["C"] * 120
    random.Random(_stable_int(DATASET_SEED, "identity")).shuffle(values)
    return values


def synthetic_users() -> tuple[_SyntheticUser, ...]:
    groups = _identity_assignments()
    rows: list[_SyntheticUser] = []
    for index in range(1, N_USERS + 1):
        rng = random.Random(_stable_int(DATASET_SEED, "user", index))
        ocean = [rng.betavariate(2.2, 2.2) for _ in range(5)]
        o, c, e, a, n = ocean
        # Personality is only partially informative: each preference dimension
        # contains an independent latent component. Identity group is generated
        # separately and never enters this function.
        preference = (
            _clip(0.65 * o + 0.20 * e + 0.15 * rng.random()),
            _clip(0.65 * c + 0.20 * a + 0.15 * rng.random()),
            _clip(0.65 * e + 0.20 * a + 0.15 * rng.random()),
            _clip(0.55 * (1.0 - n) + 0.25 * c + 0.20 * rng.random()),
            _clip(0.55 * o + 0.25 * (1.0 - n) + 0.20 * rng.random()),
        )
        rows.append(
            _SyntheticUser(
                user_id=f"u{index:03d}",
                identity_group=groups[index - 1],
                personality=PersonalityProfile(
                    openness=o,
                    conscientiousness=c,
                    extraversion=e,
                    agreeableness=a,
                    neuroticism=n,
                ),
                preference_vector=preference,
            )
        )
    return tuple(rows)


def synthetic_items() -> tuple[_SyntheticItem, ...]:
    rows: list[_SyntheticItem] = []
    for index in range(1, N_ITEMS + 1):
        rng = random.Random(_stable_int(DATASET_SEED, "item", index))
        features = tuple(rng.betavariate(1.7, 1.7) for _ in FEATURE_NAMES)
        rows.append(
            _SyntheticItem(
                item_id=f"i{index:03d}",
                title=f"FairSynth Item {index:03d}",
                features=features,
            )
        )
    return tuple(rows)


def _utility(user: _SyntheticUser, item: _SyntheticItem) -> float:
    return sum(a * b for a, b in zip(user.preference_vector, item.features, strict=True)) / len(FEATURE_NAMES)


def _item(row: _SyntheticItem) -> Item:
    return Item(
        item_id=row.item_id,
        title=row.title,
        metadata={name: round(value, 6) for name, value in zip(FEATURE_NAMES, row.features, strict=True)},
    )


class FairSynth360Adapter(DatasetAdapter):
    """Fully synthetic controlled recommendation benchmark generated in-repo.

    The synthetic identity labels A/B/C are balanced, semantically meaningless,
    and generated independently from both preference and relevance. Therefore a
    model has no ground-truth reason to use identity when ranking. Synthetic OCEAN
    values are partially predictive of the latent preference vector, allowing a
    controlled personality-value sanity test without pretending these are human
    psychometric measurements.
    """

    dataset_id = "fairsynth360"

    def __init__(self, *, relevant_items: int = 5) -> None:
        if relevant_items <= 0:
            raise ValueError("relevant_items must be positive")
        self.relevant_items = int(relevant_items)
        self._manifest: dict[str, object] = {"dataset_id": self.dataset_id}

    def card(self) -> DatasetCard:
        return DatasetCard(
            dataset_id=self.dataset_id,
            source="generated_in_repository",
            version="FairSynth-360 v1 seed=1729",
            license="project-generated; release license must be frozen before publication",
            domain="synthetic_multi_attribute_items",
            track="synthetic_stress_test",
            observed_fields=(
                "known_latent_preference_vector",
                "synthetic_ocean_context",
                "balanced_semantically_meaningless_identity_group",
            ),
            derived_fields=("history", "candidate_set", "known_relevance"),
            counterfactual_fields=("synthetic_identity_group", "synthetic_ocean"),
            notes=(
                "No external or personal data are used. Identity A/B/C is independent of "
                "ground-truth relevance. Synthetic OCEAN is not described as measured human "
                "personality and is used only for stress-testing/validation."
            ),
        )

    def build_instances(
        self,
        raw_dir: Path,
        *,
        users: int,
        candidate_set_size: int,
        max_history_items: int,
        seed: int,
    ) -> Iterator[UserInstance]:
        del raw_dir  # Dataset is generated entirely from the versioned specification.
        if users <= 0 or users > N_USERS:
            raise ValueError(f"users must be in [1,{N_USERS}]")
        if candidate_set_size < self.relevant_items + 1 or candidate_set_size > 60:
            raise ValueError(
                f"candidate_set_size must be between {self.relevant_items + 1} and 60"
            )
        if max_history_items <= 0 or max_history_items > 60:
            raise ValueError("max_history_items must be in [1,60]")

        all_users = synthetic_users()
        all_items = synthetic_items()
        history_pool = all_items[:60]
        candidate_pool = all_items[60:]

        selected = sorted(
            all_users,
            key=lambda user: _stable_int(seed, self.dataset_id, "select", user.user_id),
        )[:users]

        emitted = 0
        for user in selected:
            history_ranked = sorted(
                history_pool,
                key=lambda item: (-_utility(user, item), item.item_id),
            )
            history = [_item(item) for item in history_ranked[:max_history_items]]

            candidate_ranked = sorted(
                candidate_pool,
                key=lambda item: (-_utility(user, item), item.item_id),
            )
            relevant = candidate_ranked[: self.relevant_items]
            relevant_ids = {item.item_id for item in relevant}
            remaining = [item for item in candidate_pool if item.item_id not in relevant_ids]
            random.Random(_stable_int(seed, self.dataset_id, "negatives", user.user_id)).shuffle(remaining)
            chosen = list(relevant) + remaining[: candidate_set_size - len(relevant)]
            random.Random(_stable_int(seed, self.dataset_id, "candidate_order", user.user_id)).shuffle(chosen)

            instance = UserInstance(
                dataset=self.dataset_id,
                user_id=user.user_id,
                history=history,
                candidates=[_item(item) for item in chosen],
                relevant_item_ids=frozenset(relevant_ids),
                demographics={"synthetic_identity_group": user.identity_group},
                personality=user.personality,
                instance_metadata={
                    "synthetic": True,
                    "dataset_version": "FairSynth-360-v1",
                    "dataset_seed": DATASET_SEED,
                    "generation_seed": int(seed),
                    "identity_semantics": "balanced_semantically_meaningless_independent_of_relevance",
                    "personality_instrument": "synthetic_OCEAN_v1_not_human_measurement",
                    "personality_score_view": "generated_controlled_0_1",
                    "personality_raw_scale": "0_to_1",
                    "latent_preference_vector": [round(value, 6) for value in user.preference_vector],
                    "ground_truth_relevance_policy": "top_latent_utility_within_candidate_pool",
                },
            )
            instance.validate()
            emitted += 1
            yield instance

        self._manifest = {
            "dataset_id": self.dataset_id,
            "version": "FairSynth-360-v1",
            "dataset_seed": DATASET_SEED,
            "generation_seed": int(seed),
            "available_users": N_USERS,
            "catalog_items": N_ITEMS,
            "identity_groups": {"A": 120, "B": 120, "C": 120},
            "identity_independent_of_relevance_by_construction": True,
            "synthetic_personality_is_human_measurement": False,
            "users_emitted": emitted,
            "candidate_set_size": int(candidate_set_size),
            "max_history_items": int(max_history_items),
            "relevant_items": self.relevant_items,
        }

    def preprocessing_manifest(self) -> dict[str, object]:
        return dict(self._manifest)
