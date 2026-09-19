from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterator

from .base import DatasetAdapter, DatasetCard
from ..schema import Item, PersonalityProfile, UserInstance


BIG_FIVE_ITEMS = {
    "neuroticism": (7, 11, 12),
    "conscientiousness": (6, 8, 15),
    "agreeableness": (1, 9, 13),
    "openness": (3, 4, 10),
    "extraversion": (2, 5, 14),
}
REVERSE_SCORED = {2, 5}


def _stable_int(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return [dict(row) for row in reader]


def score_cbf_pi_15(row: dict[str, str]) -> PersonalityProfile:
    """Score REASONER's CBF-PI-15 responses into normalized Big Five means.

    The official README specifies a 0--5 six-point scale and reverse scoring for
    Q2 and Q5. Each dimension contains exactly three items. Means are divided by
    five to obtain the [0,1] representation used by FairEval prompts.
    """
    values: dict[int, float] = {}
    for item in range(1, 16):
        key = f"Q{item}"
        if key not in row or row[key] in (None, ""):
            raise ValueError(f"missing {key} in CBF-PI-15 response")
        value = float(row[key])
        if not 0.0 <= value <= 5.0:
            raise ValueError(f"{key} must be in [0,5], got {value}")
        if item in REVERSE_SCORED:
            value = 5.0 - value
        values[item] = value

    scores = {
        dimension: sum(values[item] for item in items) / (len(items) * 5.0)
        for dimension, items in BIG_FIVE_ITEMS.items()
    }
    return PersonalityProfile(
        openness=scores["openness"],
        conscientiousness=scores["conscientiousness"],
        extraversion=scores["extraversion"],
        agreeableness=scores["agreeableness"],
        neuroticism=scores["neuroticism"],
    )


class ReasonerAdapter(DatasetAdapter):
    """Adapter for the official REASONER recommendation dataset.

    REASONER does not provide interaction timestamps in the documented schema.
    FairEval therefore uses a deterministic per-user random split rather than
    pretending the rows are chronological. The split policy is logged explicitly.
    """

    dataset_id = "reasoner"

    def __init__(
        self,
        *,
        positive_threshold: float = 4.0,
        train_fraction: float = 0.8,
        max_relevant_per_user: int = 5,
        min_interactions: int = 8,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        self.positive_threshold = float(positive_threshold)
        self.train_fraction = float(train_fraction)
        self.max_relevant_per_user = int(max_relevant_per_user)
        self.min_interactions = int(min_interactions)
        self._manifest: dict[str, object] = {"dataset_id": self.dataset_id}

    def card(self) -> DatasetCard:
        return DatasetCard(
            dataset_id=self.dataset_id,
            source="https://github.com/REASONER2023/reasoner2023.github.io",
            version="official REASONER dataset release",
            license="CC BY-NC 4.0 plus upstream terms",
            domain="short_video",
            track="personality_grounded",
            observed_fields=(
                "interactions",
                "ratings",
                "likes",
                "user_profile",
                "video_metadata",
                "CBF_PI_15_responses",
            ),
            derived_fields=("normalized_big_five", "train_test_split", "candidate_set"),
            counterfactual_fields=("big_five_one_trait",),
            notes=(
                "CBF-PI-15 uses 15 items on a 0-5 scale; Q2 and Q5 are reverse-scored. "
                "FairEval uses the measured personality track and does not infer traits."
            ),
        )

    def _load(self, raw_dir: Path):
        interaction_path = raw_dir / "interaction.csv"
        user_path = raw_dir / "user.csv"
        video_path = raw_dir / "video.csv"
        bigfive_path = raw_dir / "bigfive.csv"
        for path in (interaction_path, user_path, video_path, bigfive_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing REASONER file: {path}")

        interactions = _read_tsv(interaction_path)
        users = {row["user_id"]: row for row in _read_tsv(user_path)}
        videos = {row["video_id"]: row for row in _read_tsv(video_path)}
        personality = {
            row["user_id"]: score_cbf_pi_15(row)
            for row in _read_tsv(bigfive_path)
        }

        by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
        popularity: Counter[str] = Counter()
        for row in interactions:
            user_id = row.get("user_id", "")
            video_id = row.get("video_id", "")
            if user_id not in users or user_id not in personality or video_id not in videos:
                continue
            by_user[user_id].append(row)
            popularity[video_id] += 1

        self._manifest = {
            "dataset_id": self.dataset_id,
            "source_version": "official REASONER dataset release",
            "license": "CC BY-NC 4.0 plus upstream terms",
            "interactions_loaded": len(interactions),
            "users_loaded": len(users),
            "videos_loaded": len(videos),
            "personality_profiles_loaded": len(personality),
            "personality_instrument": "CBF-PI-15",
            "personality_raw_scale": "0_to_5",
            "reverse_scored_items": [2, 5],
            "positive_threshold": self.positive_threshold,
            "train_fraction": self.train_fraction,
            "split_policy": "deterministic_per_user_random",
        }
        return users, videos, personality, by_user, popularity

    @staticmethod
    def _video_item(video_id: str, videos: dict[str, dict[str, str]], **extra) -> Item:
        row = videos[video_id]
        metadata: dict[str, object] = {
            "category": row.get("category", ""),
            "tags": row.get("tags", ""),
        }
        metadata.update(extra)
        return Item(video_id, row.get("title", f"video_{video_id}"), metadata)

    def build_instances(
        self,
        raw_dir: Path,
        *,
        users: int,
        candidate_set_size: int,
        max_history_items: int,
        seed: int,
    ) -> Iterator[UserInstance]:
        if users <= 0:
            raise ValueError("users must be positive")
        if candidate_set_size < 2:
            raise ValueError("candidate_set_size must be >= 2")
        if max_history_items <= 0:
            raise ValueError("max_history_items must be positive")

        profiles, videos, personality, by_user, popularity = self._load(raw_dir)
        all_video_ids = tuple(sorted(videos, key=lambda x: int(x)))

        prepared: dict[str, tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]] = {}
        eligible: list[tuple[int, str]] = []
        for user_id, rows in by_user.items():
            if len(rows) < self.min_interactions:
                continue
            shuffled = list(rows)
            random.Random(_stable_int(seed, self.dataset_id, "split", user_id)).shuffle(shuffled)
            split = max(1, min(len(shuffled) - 1, int(math.floor(len(shuffled) * self.train_fraction))))
            train = shuffled[:split]
            test = shuffled[split:]
            relevant = [
                row
                for row in test
                if float(row.get("rating") or 0.0) >= self.positive_threshold
            ]
            if not relevant:
                continue
            relevant = relevant[: self.max_relevant_per_user]
            prepared[user_id] = (train, test, relevant)
            eligible.append((_stable_int(seed, self.dataset_id, "user", user_id), user_id))

        eligible.sort()
        selected_ids = [user_id for _, user_id in eligible[:users]]
        if not selected_ids:
            raise ValueError("no eligible REASONER users after filtering")

        emitted = 0
        for user_id in selected_ids:
            train, test, relevant = prepared[user_id]
            relevant_ids = [row["video_id"] for row in relevant]
            if len(relevant_ids) >= candidate_set_size:
                relevant_ids = relevant_ids[: candidate_set_size - 1]
            relevant_set = set(relevant_ids)

            interacted_ids = {row["video_id"] for row in by_user[user_id]}
            negative_ids: list[str] = []
            for row in test:
                video_id = row["video_id"]
                if video_id in relevant_set:
                    continue
                if float(row.get("rating") or 0.0) < self.positive_threshold and video_id not in negative_ids:
                    negative_ids.append(video_id)
                if len(relevant_ids) + len(negative_ids) >= candidate_set_size:
                    break

            target_popularity = median(popularity[vid] for vid in relevant_ids)
            unseen = [
                vid for vid in all_video_ids
                if vid not in interacted_ids and vid not in negative_ids
            ]
            unseen.sort(
                key=lambda vid: (
                    abs(math.log1p(popularity[vid]) - math.log1p(target_popularity)),
                    _stable_int(seed, user_id, vid),
                )
            )
            needed = candidate_set_size - len(relevant_ids) - len(negative_ids)
            negative_ids.extend(unseen[:needed])
            candidate_ids = relevant_ids + negative_ids
            if len(candidate_ids) < candidate_set_size:
                continue

            random.Random(_stable_int(seed, "candidate_order", user_id)).shuffle(candidate_ids)

            history_rows = train[-max_history_items:]
            history = [
                self._video_item(
                    row["video_id"],
                    videos,
                    rating=float(row.get("rating") or 0.0),
                    liked=int(row.get("like") or 0),
                )
                for row in history_rows
            ]
            candidates = [self._video_item(video_id, videos) for video_id in candidate_ids]

            user_row = profiles[user_id]
            gender_code = user_row.get("gender", "")
            gender = {"0": "female", "1": "male"}.get(gender_code, f"code_{gender_code}")
            age_code = user_row.get("age", "")

            instance = UserInstance(
                dataset=self.dataset_id,
                user_id=user_id,
                history=history,
                candidates=candidates,
                relevant_item_ids=frozenset(relevant_set),
                demographics={
                    "gender": gender,
                    "age_code": f"code_{age_code}",
                },
                personality=personality[user_id],
                instance_metadata={
                    "split_policy": "deterministic_per_user_random",
                    "positive_threshold": self.positive_threshold,
                    "candidate_policy": "heldout_nonrelevant_then_popularity_matched_unseen",
                    "candidate_seed": seed,
                    "personality_instrument": "CBF-PI-15",
                    "personality_raw_scale": "0_to_5",
                    "personality_score_view": "dimension_mean_normalized_0_1",
                    "personality_reverse_scored_items": [2, 5],
                    "demographics_secondary_in_personality_track": True,
                },
            )
            instance.validate()
            emitted += 1
            yield instance

        self._manifest = {
            **self._manifest,
            "eligible_users": len(eligible),
            "requested_users": users,
            "emitted_users": emitted,
            "candidate_set_size": candidate_set_size,
            "max_history_items": max_history_items,
            "seed": seed,
            "user_selection_hash": hashlib.sha256(
                "\n".join(selected_ids).encode("utf-8")
            ).hexdigest(),
        }

    def preprocessing_manifest(self) -> dict[str, object]:
        return dict(self._manifest)
