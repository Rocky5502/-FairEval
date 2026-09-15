from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterator

import pandas as pd

from .base import DatasetAdapter, DatasetCard
from ..schema import Item, PersonalityProfile, UserInstance


def _stable_int(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _norm_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _resolve_column(columns: list[str], aliases: tuple[str, ...], *, field: str) -> str:
    by_norm: dict[str, list[str]] = defaultdict(list)
    for column in columns:
        by_norm[_norm_name(column)].append(column)

    hits: list[str] = []
    for alias in aliases:
        hits.extend(by_norm.get(_norm_name(alias), []))
    hits = list(dict.fromkeys(hits))
    if len(hits) != 1:
        raise ValueError(
            f"Music Master schema must contain exactly one column for {field}; "
            f"aliases={aliases!r}, matches={hits!r}, columns={columns!r}"
        )
    return hits[0]


def _optional_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    by_norm = {_norm_name(column): column for column in columns}
    for alias in aliases:
        hit = by_norm.get(_norm_name(alias))
        if hit is not None:
            return hit
    return None


def _as_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _scale(value: object, *, field: str, lo: float, hi: float) -> float:
    score = _as_float(value, field=field)
    if not lo <= score <= hi:
        raise ValueError(
            f"{field}={score} lies outside declared BFI-2 scale [{lo},{hi}]. "
            "Do not auto-rescale an unknown source representation; inspect the raw file "
            "and configure the correct scale explicitly."
        )
    return (score - lo) / (hi - lo)


class MusicMasterBFI2Adapter(DatasetAdapter):
    """Adapter for the public Music Master / BFI-2 recommendation dataset.

    The associated study reports 279 participants, 745 songs, 5,278 user-song
    rows, three 1--5 rating types (Q1/Q2/Q3), 20 BFI-2 personality dimensions,
    and 29 audio features. The public Figshare artifact is distributed as an
    Excel workbook. This adapter also accepts a CSV export of the same table.

    We deliberately use Q1 ("how much do you like this song?") as the primary
    held-out relevance signal. Q2 and Q3 are retained as metadata for planned
    sensitivity analyses; they must not be silently substituted for Q1.
    """

    dataset_id = "music_master_bfi2"

    USER_ALIASES = (
        "user_id",
        "userid",
        "user",
        "participant_id",
        "participantid",
        "participant",
        "listener_id",
        "listenerid",
    )
    ITEM_ALIASES = (
        "song_id",
        "songid",
        "song",
        "track_id",
        "trackid",
        "track",
        "music_id",
        "musicid",
        "file_id",
        "fileid",
        "filename",
    )
    Q1_ALIASES = ("q1", "rating_q1", "rating1", "like", "liking", "preference")
    Q2_ALIASES = ("q2", "rating_q2", "rating2")
    Q3_ALIASES = ("q3", "rating_q3", "rating3")

    TRAIT_ALIASES = {
        "openness": (
            "openness",
            "open_mindedness",
            "openmindedness",
            "open-mindedness",
            "o",
        ),
        "conscientiousness": ("conscientiousness", "conscientious", "c"),
        "extraversion": ("extraversion", "extroversion", "extravert", "e"),
        "agreeableness": ("agreeableness", "agreeable", "a"),
        "neuroticism": (
            "neuroticism",
            "negative_emotionality",
            "negativeemotionality",
            "n",
        ),
    }

    def __init__(
        self,
        *,
        positive_threshold: float = 4.0,
        train_fraction: float = 0.8,
        max_relevant_per_user: int = 5,
        min_interactions: int = 10,
        personality_scale_min: float = 1.0,
        personality_scale_max: float = 5.0,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if personality_scale_max <= personality_scale_min:
            raise ValueError("personality scale max must exceed min")
        self.positive_threshold = float(positive_threshold)
        self.train_fraction = float(train_fraction)
        self.max_relevant_per_user = int(max_relevant_per_user)
        self.min_interactions = int(min_interactions)
        self.personality_scale_min = float(personality_scale_min)
        self.personality_scale_max = float(personality_scale_max)
        self._manifest: dict[str, object] = {"dataset_id": self.dataset_id}

    def card(self) -> DatasetCard:
        return DatasetCard(
            dataset_id=self.dataset_id,
            source="https://doi.org/10.6084/m9.figshare.19678962",
            version="Figshare article 19678962; freeze downloaded file checksum before run",
            license="CC BY 4.0; associated-paper citation required by dataset page",
            domain="music",
            track="personality_grounded",
            observed_fields=(
                "song_ratings_q1_q2_q3",
                "bfi2_five_domains_and_15_facets",
                "audio_features_29d",
            ),
            derived_fields=("normalized_ocean", "heldout_relevance", "candidate_set"),
            counterfactual_fields=("shuffled_personality", "one_trait_personality"),
            notes=(
                "Q1 is the primary preference signal. Q2/Q3 remain secondary outcomes. "
                "The adapter accepts the original Excel file or a CSV export and refuses "
                "ambiguous schema matches."
            ),
        )

    @staticmethod
    def _locate_file(raw_dir: Path) -> Path:
        preferred = [
            raw_dir / "music_master.xlsx",
            raw_dir / "MusicMaster.xlsx",
            raw_dir / "music_master_bfi2.xlsx",
            raw_dir / "music_master.csv",
            raw_dir / "MusicMaster.csv",
            raw_dir / "music_master_bfi2.csv",
        ]
        for path in preferred:
            if path.is_file():
                return path

        candidates = sorted(
            path for path in raw_dir.iterdir() if path.suffix.lower() in {".xlsx", ".xls", ".csv"}
        ) if raw_dir.is_dir() else []
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(
            "Music Master raw directory must contain one identifiable .xlsx/.xls/.csv file; "
            f"found {[p.name for p in candidates]!r}"
        )

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            try:
                frame = pd.read_excel(path)
            except ImportError as exc:
                raise ImportError(
                    "Reading the original Music Master Excel workbook requires openpyxl. "
                    "Install the project dependencies or export the workbook to CSV."
                ) from exc
        if frame.empty:
            raise ValueError(f"{path} contains no rows")
        frame.columns = [str(column).strip() for column in frame.columns]
        return frame

    def _schema(self, frame: pd.DataFrame) -> dict[str, str | None]:
        columns = list(frame.columns)
        schema: dict[str, str | None] = {
            "user": _resolve_column(columns, self.USER_ALIASES, field="user_id"),
            "item": _resolve_column(columns, self.ITEM_ALIASES, field="song_id"),
            "q1": _resolve_column(columns, self.Q1_ALIASES, field="Q1"),
            "q2": _optional_column(columns, self.Q2_ALIASES),
            "q3": _optional_column(columns, self.Q3_ALIASES),
            "title": _optional_column(columns, ("title", "song_title", "track_title", "name")),
            "artist": _optional_column(columns, ("artist", "artist_name", "performer")),
            "genre": _optional_column(columns, ("genre", "music_genre", "style")),
        }
        for trait, aliases in self.TRAIT_ALIASES.items():
            schema[trait] = _resolve_column(columns, aliases, field=trait)
        return schema

    def _profile(self, row: pd.Series, schema: dict[str, str | None]) -> PersonalityProfile:
        values: dict[str, float] = {}
        for trait in self.TRAIT_ALIASES:
            column = schema[trait]
            assert column is not None
            values[trait] = _scale(
                row[column],
                field=f"{trait}/{column}",
                lo=self.personality_scale_min,
                hi=self.personality_scale_max,
            )
        return PersonalityProfile(**values)

    @staticmethod
    def _stable_row_key(row: dict[str, object]) -> tuple[str, str]:
        return (str(row["item_id"]), str(row["row_index"]))

    def _load(self, raw_dir: Path):
        path = self._locate_file(raw_dir)
        frame = self._read(path)
        schema = self._schema(frame)

        by_user: dict[str, list[dict[str, object]]] = defaultdict(list)
        profiles: dict[str, PersonalityProfile] = {}
        item_metadata: dict[str, dict[str, object]] = {}
        popularity: Counter[str] = Counter()

        for row_index, (_, row) in enumerate(frame.iterrows()):
            user_col = schema["user"]
            item_col = schema["item"]
            q1_col = schema["q1"]
            assert user_col is not None and item_col is not None and q1_col is not None
            user_id = str(row[user_col]).strip()
            item_id = str(row[item_col]).strip()
            if not user_id or not item_id or user_id.lower() == "nan" or item_id.lower() == "nan":
                continue

            profile = self._profile(row, schema)
            previous = profiles.get(user_id)
            if previous is not None and previous != profile:
                raise ValueError(
                    f"inconsistent BFI-2 profile across rows for user {user_id!r}; "
                    "inspect the source workbook instead of averaging silently"
                )
            profiles[user_id] = profile

            q1 = _as_float(row[q1_col], field="Q1")
            if not 1.0 <= q1 <= 5.0:
                raise ValueError(f"Q1 must be on the documented 1--5 scale, got {q1}")
            q2 = None
            q3 = None
            if schema["q2"] is not None and not pd.isna(row[schema["q2"]]):
                q2 = _as_float(row[schema["q2"]], field="Q2")
            if schema["q3"] is not None and not pd.isna(row[schema["q3"]]):
                q3 = _as_float(row[schema["q3"]], field="Q3")

            metadata: dict[str, object] = {}
            for key in ("title", "artist", "genre"):
                column = schema[key]
                if column is not None and not pd.isna(row[column]):
                    metadata[key] = str(row[column]).strip()
            if item_id not in item_metadata:
                item_metadata[item_id] = metadata

            by_user[user_id].append(
                {
                    "row_index": row_index,
                    "item_id": item_id,
                    "q1": q1,
                    "q2": q2,
                    "q3": q3,
                }
            )
            popularity[item_id] += 1

        if not profiles or not by_user:
            raise ValueError("Music Master adapter produced no usable users")

        semantic_fields = sorted(
            key for key in ("title", "artist", "genre") if any(key in m for m in item_metadata.values())
        )
        self._manifest = {
            "dataset_id": self.dataset_id,
            "source_file": path.name,
            "rows_loaded": int(sum(len(rows) for rows in by_user.values())),
            "users_loaded": len(by_user),
            "items_loaded": len(item_metadata),
            "resolved_schema": {key: value for key, value in schema.items()},
            "primary_rating": "Q1",
            "positive_threshold": self.positive_threshold,
            "personality_instrument": "BFI-2",
            "personality_raw_scale": f"{self.personality_scale_min}_to_{self.personality_scale_max}",
            "personality_normalization": "(x-min)/(max-min)",
            "semantic_item_fields_available": semantic_fields,
            "train_fraction": self.train_fraction,
        }
        return profiles, item_metadata, by_user, popularity

    @staticmethod
    def _item(
        item_id: str,
        item_metadata: dict[str, dict[str, object]],
        *,
        q1: float | None = None,
        q2: float | None = None,
        q3: float | None = None,
    ) -> Item:
        source = dict(item_metadata.get(item_id, {}))
        title = str(source.pop("title", item_id))
        if q1 is not None:
            source["q1_rating"] = float(q1)
        if q2 is not None:
            source["q2_rating"] = float(q2)
        if q3 is not None:
            source["q3_rating"] = float(q3)
        return Item(item_id=item_id, title=title, metadata=source)

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

        profiles, item_metadata, by_user, popularity = self._load(raw_dir)
        all_item_ids = tuple(sorted(item_metadata))

        prepared: dict[str, tuple[list[dict[str, object]], list[dict[str, object]]]] = {}
        eligible: list[tuple[int, str]] = []
        for user_id, rows in by_user.items():
            if len(rows) < self.min_interactions:
                continue
            ordered = sorted(rows, key=lambda row: _stable_int(seed, "row", user_id, row["item_id"], row["row_index"]))
            split = max(1, min(len(ordered) - 1, int(math.floor(len(ordered) * self.train_fraction))))
            train = ordered[:split]
            test = ordered[split:]
            relevant = [row for row in test if float(row["q1"]) >= self.positive_threshold]
            if not relevant:
                continue
            prepared[user_id] = (train, relevant[-self.max_relevant_per_user :])
            eligible.append((_stable_int(seed, self.dataset_id, user_id), user_id))

        eligible.sort()
        selected_ids = [user_id for _, user_id in eligible[:users]]
        if not selected_ids:
            raise ValueError("no eligible Music Master users after filtering")

        emitted = 0
        for user_id in selected_ids:
            train, relevant = prepared[user_id]
            relevant_ids = list(dict.fromkeys(str(row["item_id"]) for row in relevant))
            if len(relevant_ids) >= candidate_set_size:
                relevant_ids = relevant_ids[: candidate_set_size - 1]
            relevant_set = set(relevant_ids)
            rated_ids = {str(row["item_id"]) for row in by_user[user_id]}

            target_popularity = median(popularity[item_id] for item_id in relevant_ids)
            unseen = [item_id for item_id in all_item_ids if item_id not in rated_ids]
            unseen.sort(
                key=lambda item_id: (
                    abs(math.log1p(popularity[item_id]) - math.log1p(target_popularity)),
                    _stable_int(seed, "negative", user_id, item_id),
                )
            )
            needed = candidate_set_size - len(relevant_ids)
            candidate_ids = relevant_ids + unseen[:needed]
            if len(candidate_ids) < candidate_set_size:
                continue
            random.Random(_stable_int(seed, "candidate_order", user_id)).shuffle(candidate_ids)

            history_rows = train[-max_history_items:]
            history = [
                self._item(
                    str(row["item_id"]),
                    item_metadata,
                    q1=float(row["q1"]),
                    q2=None if row["q2"] is None else float(row["q2"]),
                    q3=None if row["q3"] is None else float(row["q3"]),
                )
                for row in history_rows
            ]
            candidates = [self._item(item_id, item_metadata) for item_id in candidate_ids]

            instance = UserInstance(
                dataset=self.dataset_id,
                user_id=user_id,
                history=history,
                candidates=candidates,
                relevant_item_ids=frozenset(relevant_set),
                personality=profiles[user_id],
                instance_metadata={
                    "split_policy": "deterministic_user_level_hash_80_20_no_timestamp_claim",
                    "positive_threshold": self.positive_threshold,
                    "primary_rating": "Q1",
                    "candidate_policy": "heldout_positive_plus_popularity_matched_unseen",
                    "candidate_seed": seed,
                    "personality_instrument": "BFI-2",
                    "personality_raw_scale": f"{self.personality_scale_min}_to_{self.personality_scale_max}",
                    "personality_score_view": "normalized_0_1_bfi2_domains",
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
