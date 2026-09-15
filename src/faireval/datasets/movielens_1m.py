from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterator

from .base import DatasetAdapter, DatasetCard
from ..schema import Item, UserInstance


AGE_LABELS = {
    "1": "under_18",
    "18": "18_24",
    "25": "25_34",
    "35": "35_44",
    "45": "45_49",
    "50": "50_55",
    "56": "56_plus",
}

GENDER_LABELS = {
    "M": "male",
    "F": "female",
}


@dataclass(frozen=True)
class _Rating:
    user_id: str
    movie_id: str
    rating: float
    timestamp: int


def _stable_int(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _read_double_colon(path: Path, expected_min_fields: int) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="latin-1") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n\r")
            if not line:
                continue
            fields = line.split("::")
            if len(fields) < expected_min_fields:
                raise ValueError(
                    f"{path.name}:{line_no}: expected >= {expected_min_fields} fields, "
                    f"got {len(fields)}"
                )
            rows.append(fields)
    return rows


class MovieLens1MAdapter(DatasetAdapter):
    """Build deterministic candidate-ranking tasks from MovieLens 1M.

    Expected raw files are the official ``users.dat``, ``movies.dat``, and
    ``ratings.dat`` files. No network download is performed by the adapter.

    Design choices
    --------------
    * chronology defines history vs evaluation suffix;
    * ratings >= ``positive_threshold`` are relevant;
    * held-out low-rated movies are used first as hard negatives;
    * remaining negatives are unseen movies matched approximately by popularity;
    * demographic values come only from ``users.dat`` and are never inferred;
    * ZIP code is intentionally not exposed as a fairness attribute.
    """

    dataset_id = "movielens_1m"

    def __init__(
        self,
        *,
        positive_threshold: float = 4.0,
        train_fraction: float = 0.8,
        max_relevant_per_user: int = 5,
        min_interactions: int = 20,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if max_relevant_per_user <= 0:
            raise ValueError("max_relevant_per_user must be positive")
        self.positive_threshold = float(positive_threshold)
        self.train_fraction = float(train_fraction)
        self.max_relevant_per_user = int(max_relevant_per_user)
        self.min_interactions = int(min_interactions)
        self._manifest: dict[str, object] = {"dataset_id": self.dataset_id}

    def card(self) -> DatasetCard:
        return DatasetCard(
            dataset_id=self.dataset_id,
            source="https://grouplens.org/datasets/movielens/1m/",
            version="MovieLens-1M official release",
            license="MovieLens upstream terms; verify before redistribution",
            domain="movies",
            track="demographic_counterfactual",
            observed_fields=("ratings", "gender", "age_group", "occupation", "movie_genres"),
            derived_fields=("chronological_history", "heldout_relevance", "candidate_set"),
            counterfactual_fields=("gender", "age_group"),
            notes=(
                "ZIP code is present upstream but excluded from FairEval prompt context. "
                "Gender/age labels are dataset-provided historical fields and should not "
                "be interpreted as comprehensive contemporary identity categories."
            ),
        )

    def _load(
        self,
        raw_dir: Path,
    ) -> tuple[
        dict[str, dict[str, str]],
        dict[str, tuple[str, tuple[str, ...]]],
        dict[str, list[_Rating]],
        Counter[str],
    ]:
        users_path = raw_dir / "users.dat"
        movies_path = raw_dir / "movies.dat"
        ratings_path = raw_dir / "ratings.dat"
        for path in (users_path, movies_path, ratings_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing {path}; place the official MovieLens 1M files in {raw_dir}"
                )

        users: dict[str, dict[str, str]] = {}
        for fields in _read_double_colon(users_path, 5):
            user_id, gender, age_code, occupation, _zip_code = fields[:5]
            users[user_id] = {
                "gender": GENDER_LABELS.get(gender, gender.lower()),
                "age_group": AGE_LABELS.get(age_code, f"code_{age_code}"),
                "occupation": occupation,
            }

        movies: dict[str, tuple[str, tuple[str, ...]]] = {}
        for fields in _read_double_colon(movies_path, 3):
            movie_id, title, genres = fields[:3]
            movies[movie_id] = (title, tuple(g for g in genres.split("|") if g))

        by_user: dict[str, list[_Rating]] = defaultdict(list)
        popularity: Counter[str] = Counter()
        for fields in _read_double_colon(ratings_path, 4):
            user_id, movie_id, rating, timestamp = fields[:4]
            if user_id not in users or movie_id not in movies:
                continue
            record = _Rating(user_id, movie_id, float(rating), int(timestamp))
            by_user[user_id].append(record)
            popularity[movie_id] += 1

        for ratings in by_user.values():
            ratings.sort(key=lambda x: (x.timestamp, int(x.movie_id)))

        self._manifest = {
            "dataset_id": self.dataset_id,
            "source_version": "MovieLens-1M official release",
            "users_loaded": len(users),
            "movies_loaded": len(movies),
            "ratings_loaded": sum(len(v) for v in by_user.values()),
            "positive_threshold": self.positive_threshold,
            "train_fraction": self.train_fraction,
            "max_relevant_per_user": self.max_relevant_per_user,
            "min_interactions": self.min_interactions,
            "demographic_fields_exposed": ["gender", "age_group"],
            "demographic_fields_observed_but_not_primary": ["occupation"],
            "fields_intentionally_excluded": ["zip_code"],
        }
        return users, movies, by_user, popularity

    @staticmethod
    def _item(
        movie_id: str,
        movies: dict[str, tuple[str, tuple[str, ...]]],
        *,
        rating: float | None = None,
    ) -> Item:
        title, genres = movies[movie_id]
        metadata: dict[str, object] = {"genres": list(genres)}
        if rating is not None:
            metadata["rating"] = float(rating)
        return Item(movie_id, title, metadata)

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

        profiles, movies, by_user, popularity = self._load(raw_dir)
        all_movie_ids = tuple(sorted(movies, key=lambda x: int(x)))

        eligible: list[tuple[int, str]] = []
        prepared: dict[str, tuple[list[_Rating], list[_Rating], list[_Rating]]] = {}
        for user_id, ratings in by_user.items():
            if len(ratings) < self.min_interactions:
                continue
            split = max(1, min(len(ratings) - 1, int(math.floor(len(ratings) * self.train_fraction))))
            train = ratings[:split]
            test = ratings[split:]
            relevant = [r for r in test if r.rating >= self.positive_threshold]
            if not relevant:
                continue
            relevant = relevant[-self.max_relevant_per_user :]
            prepared[user_id] = (train, test, relevant)
            eligible.append((_stable_int(seed, self.dataset_id, user_id), user_id))

        eligible.sort()
        selected_ids = [user_id for _, user_id in eligible[:users]]
        if not selected_ids:
            raise ValueError("no eligible MovieLens users after filtering")

        emitted = 0
        for user_id in selected_ids:
            train, test, relevant = prepared[user_id]
            relevant_ids = [r.movie_id for r in relevant]
            relevant_set = set(relevant_ids)

            if len(relevant_ids) >= candidate_set_size:
                relevant_ids = relevant_ids[: candidate_set_size - 1]
                relevant_set = set(relevant_ids)

            rated_ids = {r.movie_id for r in by_user[user_id]}
            heldout_nonrelevant = [
                r.movie_id
                for r in test
                if r.movie_id not in relevant_set and r.rating < self.positive_threshold
            ]

            # Prefer explicit held-out dislikes as hard negatives, then choose unseen
            # items with popularity closest to the relevant items. This avoids an
            # arbitrary LLM-generated candidate pool and reduces popularity mismatch.
            negative_ids: list[str] = []
            for movie_id in heldout_nonrelevant:
                if movie_id not in negative_ids:
                    negative_ids.append(movie_id)
                if len(relevant_ids) + len(negative_ids) >= candidate_set_size:
                    break

            target_popularity = median(popularity[mid] for mid in relevant_ids)
            unseen = [
                movie_id
                for movie_id in all_movie_ids
                if movie_id not in rated_ids and movie_id not in negative_ids
            ]
            unseen.sort(
                key=lambda mid: (
                    abs(math.log1p(popularity[mid]) - math.log1p(target_popularity)),
                    _stable_int(seed, user_id, mid),
                )
            )
            needed = candidate_set_size - len(relevant_ids) - len(negative_ids)
            negative_ids.extend(unseen[:needed])

            candidate_ids = relevant_ids + negative_ids
            if len(candidate_ids) < candidate_set_size:
                continue

            order_rng = random.Random(_stable_int(seed, "candidate_order", user_id))
            order_rng.shuffle(candidate_ids)

            history_records = train[-max_history_items:]
            history = [
                self._item(r.movie_id, movies, rating=r.rating)
                for r in history_records
            ]
            candidates = [self._item(movie_id, movies) for movie_id in candidate_ids]

            profile = profiles[user_id]
            instance = UserInstance(
                dataset=self.dataset_id,
                user_id=user_id,
                history=history,
                candidates=candidates,
                relevant_item_ids=frozenset(relevant_set),
                demographics={
                    "gender": profile["gender"],
                    "age_group": profile["age_group"],
                },
                instance_metadata={
                    "split_policy": "chronological_80_20",
                    "positive_threshold": self.positive_threshold,
                    "candidate_policy": "heldout_dislikes_then_popularity_matched_unseen",
                    "candidate_seed": seed,
                    "history_interactions_available": len(train),
                    "test_interactions_available": len(test),
                    "occupation_observed_not_prompted": profile["occupation"],
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
