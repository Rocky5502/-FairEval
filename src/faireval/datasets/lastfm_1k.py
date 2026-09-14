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
from ..schema import Item, UserInstance


def _stable_int(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _age_group(value: str) -> str | None:
    if not value.strip():
        return None
    try:
        age = int(float(value))
    except ValueError:
        return None
    if age < 18:
        return "under_18"
    if age <= 24:
        return "18_24"
    if age <= 34:
        return "25_34"
    if age <= 44:
        return "35_44"
    if age <= 54:
        return "45_54"
    return "55_plus"


def _artist_key(mbid: str, name: str) -> str:
    mbid = mbid.strip()
    if mbid:
        return f"mbid:{mbid}"
    normalized = " ".join(name.strip().lower().split())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]
    return f"name:{digest}"


def _timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return parsed.timestamp()


class LastFM1KAdapter(DatasetAdapter):
    """Streaming adapter for the original Last.fm 1K listening dataset.

    The interaction file contains ~19M rows, so FairEval selects user profiles
    first and stores event histories only for the selected users while streaming
    once through the full file. Global artist popularity is counted for negative
    matching without retaining all raw events in memory.
    """

    dataset_id = "lastfm_1k"

    events_filename = "userid-timestamp-artid-artname-traid-traname.tsv"
    profiles_filename = "userid-profile.tsv"

    def __init__(
        self,
        *,
        train_fraction: float = 0.8,
        max_relevant_per_user: int = 5,
        min_events: int = 100,
        require_gender_and_age: bool = True,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        self.train_fraction = float(train_fraction)
        self.max_relevant_per_user = int(max_relevant_per_user)
        self.min_events = int(min_events)
        self.require_gender_and_age = bool(require_gender_and_age)
        self._manifest: dict[str, object] = {"dataset_id": self.dataset_id}

    def card(self) -> DatasetCard:
        return DatasetCard(
            dataset_id=self.dataset_id,
            source="Last.fm Dataset - 1K users (Oscar Celma / Last.fm API)",
            version="1.0, May 2010",
            license="non-commercial research use under original Last.fm dataset terms",
            domain="music_artist",
            track="demographic_counterfactual",
            observed_fields=("timestamped_listening", "gender", "age", "country", "signup"),
            derived_fields=("age_group", "artist_play_counts", "candidate_set"),
            counterfactual_fields=("gender", "age_group", "country"),
            notes=(
                "FairEval ranks artists, not individual tracks. Age group is a deterministic "
                "derivation from the observed age field. Missing demographic values are not inferred."
            ),
        )

    def _load_profiles(self, path: Path) -> dict[str, dict[str, str]]:
        profiles: dict[str, dict[str, str]] = {}
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row_no, row in enumerate(reader):
                if not row:
                    continue
                # The original profile file is commonly distributed with a header.
                if row_no == 0 and row[0].strip().lower() in {"id", "userid", "user_id"}:
                    continue
                padded = row + [""] * (5 - len(row))
                user_id, gender, age, country, signup = [value.strip() for value in padded[:5]]
                if not user_id:
                    continue
                profiles[user_id] = {
                    "gender": gender.lower(),
                    "age": age,
                    "country": country,
                    "signup": signup,
                }
        return profiles

    def _profile_eligible(self, row: dict[str, str]) -> bool:
        if not self.require_gender_and_age:
            return True
        return row.get("gender") in {"m", "f"} and _age_group(row.get("age", "")) is not None

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

        events_path = raw_dir / self.events_filename
        profiles_path = raw_dir / self.profiles_filename
        for path in (events_path, profiles_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing Last.fm 1K file: {path}")

        profiles = self._load_profiles(profiles_path)
        profile_candidates = [
            (_stable_int(seed, self.dataset_id, user_id), user_id)
            for user_id, row in profiles.items()
            if self._profile_eligible(row)
        ]
        profile_candidates.sort()

        # Select a reserve pool because a small number of profiles may not meet
        # the interaction/candidate requirements after the event scan.
        reserve_size = min(len(profile_candidates), max(users * 3, users + 50))
        reserve_ids = {user_id for _, user_id in profile_candidates[:reserve_size]}

        events_by_user: dict[str, list[tuple[float, str]]] = defaultdict(list)
        artist_popularity: Counter[str] = Counter()
        artist_names: dict[str, str] = {}
        rows_seen = 0
        malformed_rows = 0

        with events_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                rows_seen += 1
                if len(row) < 6:
                    malformed_rows += 1
                    continue
                user_id, timestamp, artist_mbid, artist_name, _track_mbid, _track_name = row[:6]
                if not user_id or not timestamp or not artist_name:
                    malformed_rows += 1
                    continue
                try:
                    ts = _timestamp(timestamp)
                except ValueError:
                    malformed_rows += 1
                    continue
                key = _artist_key(artist_mbid, artist_name)
                artist_names.setdefault(key, artist_name.strip())
                artist_popularity[key] += 1
                if user_id in reserve_ids:
                    events_by_user[user_id].append((ts, key))

        prepared: dict[str, tuple[list[tuple[float, str]], list[tuple[float, str]], list[str]]] = {}
        eligible: list[tuple[int, str]] = []
        for user_id in reserve_ids:
            events = sorted(events_by_user.get(user_id, []), key=lambda x: (x[0], x[1]))
            if len(events) < self.min_events:
                continue
            split = max(1, min(len(events) - 1, int(math.floor(len(events) * self.train_fraction))))
            train = events[:split]
            test = events[split:]
            test_counts = Counter(key for _, key in test)
            if not test_counts:
                continue
            # Held-out top artists are binary relevance labels for ranking metrics.
            relevant_ids = [
                key for key, _count in sorted(
                    test_counts.items(),
                    key=lambda pair: (-pair[1], _stable_int(seed, user_id, pair[0])),
                )[: self.max_relevant_per_user]
            ]
            prepared[user_id] = (train, test, relevant_ids)
            eligible.append((_stable_int(seed, self.dataset_id, "eligible", user_id), user_id))

        eligible.sort()
        selected_ids = [user_id for _, user_id in eligible[:users]]
        if not selected_ids:
            raise ValueError("no eligible Last.fm users after filtering")

        all_artist_ids = tuple(artist_popularity)
        emitted = 0
        for user_id in selected_ids:
            train, test, relevant_ids = prepared[user_id]
            if len(relevant_ids) >= candidate_set_size:
                relevant_ids = relevant_ids[: candidate_set_size - 1]
            relevant_set = set(relevant_ids)
            interacted = {key for _, key in train} | {key for _, key in test}

            target_popularity = median(artist_popularity[key] for key in relevant_ids)
            unseen = [key for key in all_artist_ids if key not in interacted]
            unseen.sort(
                key=lambda key: (
                    abs(math.log1p(artist_popularity[key]) - math.log1p(target_popularity)),
                    _stable_int(seed, user_id, key),
                )
            )
            needed = candidate_set_size - len(relevant_ids)
            candidate_ids = relevant_ids + unseen[:needed]
            if len(candidate_ids) < candidate_set_size:
                continue
            random.Random(_stable_int(seed, "candidate_order", user_id)).shuffle(candidate_ids)

            train_counts = Counter(key for _, key in train)
            last_seen: dict[str, float] = {}
            for ts, key in train:
                last_seen[key] = max(ts, last_seen.get(key, ts))
            history_artist_ids = [
                key for key, _count in sorted(
                    train_counts.items(),
                    key=lambda pair: (-pair[1], -last_seen[pair[0]], pair[0]),
                )[:max_history_items]
            ]
            history = [
                Item(
                    key,
                    artist_names.get(key, key),
                    {
                        "train_play_count": train_counts[key],
                        "last_seen_utc": datetime.fromtimestamp(last_seen[key]).isoformat(),
                    },
                )
                for key in history_artist_ids
            ]
            candidates = [Item(key, artist_names.get(key, key), {}) for key in candidate_ids]

            profile = profiles[user_id]
            demographics: dict[str, str] = {}
            if profile.get("gender") in {"m", "f"}:
                demographics["gender"] = {"m": "male", "f": "female"}[profile["gender"]]
            age_group = _age_group(profile.get("age", ""))
            if age_group is not None:
                demographics["age_group"] = age_group
            if profile.get("country"):
                demographics["country"] = profile["country"]

            instance = UserInstance(
                dataset=self.dataset_id,
                user_id=user_id,
                history=history,
                candidates=candidates,
                relevant_item_ids=frozenset(relevant_set),
                demographics=demographics,
                instance_metadata={
                    "split_policy": "chronological_80_20",
                    "implicit_relevance": "top_test_artists_by_play_count",
                    "candidate_policy": "popularity_matched_unseen_artists",
                    "candidate_seed": seed,
                    "recommendation_unit": "artist",
                    "observed_age_raw": profile.get("age", ""),
                    "signup_observed_not_prompted": profile.get("signup", ""),
                },
            )
            instance.validate()
            emitted += 1
            yield instance

        self._manifest = {
            "dataset_id": self.dataset_id,
            "source_version": "Last.fm Dataset - 1K users v1.0 May 2010",
            "license": "non-commercial research use under original Last.fm terms",
            "profiles_loaded": len(profiles),
            "profile_reserve_size": reserve_size,
            "event_rows_seen": rows_seen,
            "malformed_event_rows": malformed_rows,
            "unique_artists": len(artist_popularity),
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
