from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterator

from .base import DatasetAdapter, DatasetCard
from ..schema import Item, PersonalityProfile, UserInstance


def _stable_int(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _clean_row(row: dict[str, str | None]) -> dict[str, str]:
    return {
        str(key).strip(): ("" if value is None else str(value).strip())
        for key, value in row.items()
        if key is not None
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return [_clean_row(row) for row in reader]


def _pick(row: dict[str, str], *names: str) -> str:
    lower = {key.lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"none of {names!r} present in columns {tuple(row)}")


def _normalize_1_to_7(value: str, *, field: str) -> float:
    score = float(value)
    if not 1.0 <= score <= 7.0:
        raise ValueError(f"{field} must be in [1,7], got {score}")
    return (score - 1.0) / 6.0


def score_personality2018(row: dict[str, str]) -> PersonalityProfile:
    """Normalize GroupLens Personality 2018 traits to OCEAN in [0,1].

    The source file reports emotional stability rather than neuroticism. FairEval
    therefore uses ``neuroticism = 1 - normalized(emotional_stability)`` and logs
    that transformation in the preprocessing manifest.
    """
    openness = _normalize_1_to_7(_pick(row, "openness"), field="openness")
    agreeableness = _normalize_1_to_7(_pick(row, "agreeableness"), field="agreeableness")
    stability = _normalize_1_to_7(
        _pick(row, "emotional_stability", "emotional stability"),
        field="emotional_stability",
    )
    conscientiousness = _normalize_1_to_7(
        _pick(row, "conscientiousness"), field="conscientiousness"
    )
    extraversion = _normalize_1_to_7(_pick(row, "extraversion"), field="extraversion")
    return PersonalityProfile(
        openness=openness,
        conscientiousness=conscientiousness,
        extraversion=extraversion,
        agreeableness=agreeableness,
        neuroticism=1.0 - stability,
    )


def _timestamp_key(value: str) -> tuple[int, float | str]:
    stripped = value.strip()
    try:
        return (0, float(stripped))
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        return (0, parsed.timestamp())
    except ValueError:
        # ISO-like source timestamps sort lexicographically; retaining a stable
        # fallback is preferable to pretending an arbitrary row order is time.
        return (1, stripped)


class Personality2018Adapter(DatasetAdapter):
    dataset_id = "personality2018"

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
        self.positive_threshold = float(positive_threshold)
        self.train_fraction = float(train_fraction)
        self.max_relevant_per_user = int(max_relevant_per_user)
        self.min_interactions = int(min_interactions)
        self._manifest: dict[str, object] = {"dataset_id": self.dataset_id}

    def card(self) -> DatasetCard:
        return DatasetCard(
            dataset_id=self.dataset_id,
            source="https://grouplens.org/datasets/personality-2018/",
            version="GroupLens Personality 2018 official release",
            license="verify official GroupLens README before use/redistribution",
            domain="movies",
            track="personality_grounded",
            observed_fields=("movie_ratings", "measured_personality_1_to_7"),
            derived_fields=("normalized_ocean", "heldout_relevance", "candidate_set"),
            counterfactual_fields=("shuffled_personality", "one_trait_personality"),
            notes=(
                "The personality archive is joined by movie_id to a separately pinned "
                "MovieLens movies.csv metadata file. Emotional stability is inverted "
                "to obtain neuroticism after 1-7 normalization."
            ),
        )

    def _load(self, raw_dir: Path):
        ratings_path = raw_dir / "ratings.csv"
        personality_path = raw_dir / "personality-data.csv"
        movies_path = raw_dir / "movies.csv"
        for path in (ratings_path, personality_path, movies_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing {path}; personality2018 requires ratings.csv, "
                    "personality-data.csv, and a pinned MovieLens movies.csv companion"
                )

        personality_rows = _read_csv(personality_path)
        profiles: dict[str, PersonalityProfile] = {}
        for row in personality_rows:
            user_id = _pick(row, "userid", "userId", "useri")
            if user_id and user_id not in profiles:
                profiles[user_id] = score_personality2018(row)

        movie_rows = _read_csv(movies_path)
        movies: dict[str, tuple[str, tuple[str, ...]]] = {}
        for row in movie_rows:
            movie_id = _pick(row, "movieId", "movie_id")
            title = _pick(row, "title")
            genres_value = row.get("genres", row.get(" genres", ""))
            movies[movie_id] = (title, tuple(x for x in genres_value.split("|") if x))

        by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
        popularity: Counter[str] = Counter()
        dropped_missing_movie = 0
        for row in _read_csv(ratings_path):
            user_id = _pick(row, "userid", "userId", "useri")
            movie_id = _pick(row, "movie_id", "movieId")
            if user_id not in profiles:
                continue
            if movie_id not in movies:
                dropped_missing_movie += 1
                continue
            canonical = {
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": _pick(row, "rating"),
                "tstamp": _pick(row, "tstamp", "timestamp"),
            }
            by_user[user_id].append(canonical)
            popularity[movie_id] += 1

        for rows in by_user.values():
            rows.sort(key=lambda row: (_timestamp_key(row["tstamp"]), row["movie_id"]))

        self._manifest = {
            "dataset_id": self.dataset_id,
            "source_version": "GroupLens Personality 2018 official release",
            "personality_profiles_loaded": len(profiles),
            "ratings_loaded_after_movie_join": sum(len(v) for v in by_user.values()),
            "ratings_dropped_missing_movie_metadata": dropped_missing_movie,
            "movie_metadata_rows": len(movies),
            "movie_metadata_requirement": "separately pinned MovieLens movies.csv",
            "personality_instrument_view": "source five trait scores 1_to_7",
            "personality_transformation": (
                "normalize each trait as (x-1)/6; neuroticism=1-normalized emotional_stability"
            ),
            "positive_threshold": self.positive_threshold,
            "train_fraction": self.train_fraction,
        }
        return profiles, movies, by_user, popularity

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
        all_movie_ids = tuple(sorted(movies, key=lambda value: int(value)))

        prepared: dict[str, tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]] = {}
        eligible: list[tuple[int, str]] = []
        for user_id, rows in by_user.items():
            if len(rows) < self.min_interactions:
                continue
            split = max(1, min(len(rows) - 1, int(math.floor(len(rows) * self.train_fraction))))
            train = rows[:split]
            test = rows[split:]
            relevant = [
                row for row in test if float(row["rating"]) >= self.positive_threshold
            ]
            if not relevant:
                continue
            relevant = relevant[-self.max_relevant_per_user :]
            prepared[user_id] = (train, test, relevant)
            eligible.append((_stable_int(seed, self.dataset_id, user_id), user_id))

        eligible.sort()
        selected_ids = [user_id for _, user_id in eligible[:users]]
        if not selected_ids:
            raise ValueError("no eligible Personality 2018 users after filtering")

        emitted = 0
        for user_id in selected_ids:
            train, test, relevant = prepared[user_id]
            relevant_ids = [row["movie_id"] for row in relevant]
            if len(relevant_ids) >= candidate_set_size:
                relevant_ids = relevant_ids[: candidate_set_size - 1]
            relevant_set = set(relevant_ids)
            rated_ids = {row["movie_id"] for row in by_user[user_id]}

            negative_ids: list[str] = []
            for row in test:
                movie_id = row["movie_id"]
                if movie_id in relevant_set:
                    continue
                if float(row["rating"]) < self.positive_threshold and movie_id not in negative_ids:
                    negative_ids.append(movie_id)
                if len(relevant_ids) + len(negative_ids) >= candidate_set_size:
                    break

            target_popularity = median(popularity[mid] for mid in relevant_ids)
            unseen = [
                mid for mid in all_movie_ids
                if mid not in rated_ids and mid not in negative_ids
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

            random.Random(_stable_int(seed, "candidate_order", user_id)).shuffle(candidate_ids)
            history_rows = train[-max_history_items:]
            history = [
                self._item(row["movie_id"], movies, rating=float(row["rating"]))
                for row in history_rows
            ]
            candidates = [self._item(movie_id, movies) for movie_id in candidate_ids]

            instance = UserInstance(
                dataset=self.dataset_id,
                user_id=user_id,
                history=history,
                candidates=candidates,
                relevant_item_ids=frozenset(relevant_set),
                personality=profiles[user_id],
                instance_metadata={
                    "split_policy": "chronological_80_20",
                    "positive_threshold": self.positive_threshold,
                    "candidate_policy": "heldout_dislikes_then_popularity_matched_unseen",
                    "candidate_seed": seed,
                    "personality_instrument": "GroupLens Personality 2018 five-trait assessment",
                    "personality_raw_scale": "1_to_7",
                    "personality_score_view": "normalized_0_1_ocean_with_stability_inverted",
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
