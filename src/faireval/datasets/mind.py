from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path
from typing import Iterator

from .base import DatasetAdapter, DatasetCard
from ..schema import Item, UserInstance


def _stable_int(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


class MINDAdapter(DatasetAdapter):
    """Adapter for MIND training/validation behaviors with logged impressions.

    Each FairEval task corresponds to one logged impression. Candidate items come
    only from that impression; if more candidates than requested are available,
    every clicked item is retained and negatives are deterministically subsampled.
    MIND has no observed demographic labels, so any identity cue applied later is
    explicitly a synthetic stress test.
    """

    dataset_id = "mind"

    def __init__(self, *, require_history: bool = True) -> None:
        self.require_history = bool(require_history)
        self._manifest: dict[str, object] = {"dataset_id": self.dataset_id}

    def card(self) -> DatasetCard:
        return DatasetCard(
            dataset_id=self.dataset_id,
            source="https://msnews.github.io/",
            version="MIND train/validation release; exact split must be pinned in manifest",
            license="Microsoft Research/MIND upstream data terms; verify before redistribution",
            domain="news",
            track="synthetic_stress_test",
            observed_fields=("click_history", "logged_impressions", "news_metadata"),
            derived_fields=("fixed_candidate_subset",),
            counterfactual_fields=("synthetic_identity_cue_only",),
            notes=(
                "MIND does not provide observed demographic or personality identity. "
                "FairEval must never present synthetic cue experiments on MIND as observed-group fairness."
            ),
        )

    @staticmethod
    def _load_news(path: Path) -> dict[str, Item]:
        news: dict[str, Item] = {}
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) < 5:
                    continue
                news_id, category, subcategory, title, abstract = row[:5]
                if not news_id:
                    continue
                news[news_id] = Item(
                    news_id,
                    title or news_id,
                    {
                        "category": category,
                        "subcategory": subcategory,
                        # Abstract is intentionally excluded from primary prompts to
                        # keep context size controlled across impression sizes.
                        "abstract_available": bool(abstract),
                    },
                )
        return news

    @staticmethod
    def _parse_impressions(value: str) -> tuple[list[str], list[str]]:
        candidates: list[str] = []
        positives: list[str] = []
        for token in value.split():
            if "-" not in token:
                continue
            news_id, label = token.rsplit("-", 1)
            if label not in {"0", "1"}:
                continue
            candidates.append(news_id)
            if label == "1":
                positives.append(news_id)
        return candidates, positives

    def build_instances(
        self,
        raw_dir: Path,
        *,
        users: int,
        candidate_set_size: int,
        max_history_items: int,
        seed: int,
    ) -> Iterator[UserInstance]:
        # ``users`` means benchmark tasks/impressions for MIND, because a single
        # anonymous MIND user can have multiple logged impression rows.
        if users <= 0:
            raise ValueError("users/tasks must be positive")
        if candidate_set_size < 2:
            raise ValueError("candidate_set_size must be >= 2")
        if max_history_items <= 0:
            raise ValueError("max_history_items must be positive")

        news_path = raw_dir / "news.tsv"
        behaviors_path = raw_dir / "behaviors.tsv"
        for path in (news_path, behaviors_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing MIND file: {path}")

        news = self._load_news(news_path)
        eligible: list[tuple[int, tuple[str, str, str, list[str], list[str], list[str]]]] = []
        rows_seen = 0
        skipped_no_positive = 0
        skipped_small_candidate_set = 0
        skipped_missing_news = 0

        with behaviors_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                rows_seen += 1
                if len(row) < 5:
                    continue
                impression_id, user_id, timestamp, history_raw, impressions_raw = row[:5]
                history_ids = [x for x in history_raw.split() if x in news]
                if self.require_history and not history_ids:
                    continue
                candidate_ids, positive_ids = self._parse_impressions(impressions_raw)
                candidate_ids = [x for x in candidate_ids if x in news]
                positive_ids = [x for x in positive_ids if x in news]
                if not positive_ids:
                    skipped_no_positive += 1
                    continue
                if len(candidate_ids) < candidate_set_size:
                    skipped_small_candidate_set += 1
                    continue
                if len(candidate_ids) != len(set(candidate_ids)):
                    # Duplicate impression entries would make ranking membership
                    # ambiguous; skip rather than silently deduplicate labels.
                    continue
                if any(x not in news for x in positive_ids):
                    skipped_missing_news += 1
                    continue
                payload = (
                    impression_id,
                    user_id,
                    timestamp,
                    history_ids,
                    candidate_ids,
                    positive_ids,
                )
                eligible.append((_stable_int(seed, self.dataset_id, impression_id), payload))

        eligible.sort(key=lambda x: x[0])
        chosen = eligible[:users]
        if not chosen:
            raise ValueError("no eligible MIND impressions after filtering")

        emitted = 0
        chosen_ids: list[str] = []
        for _, payload in chosen:
            impression_id, user_id, timestamp, history_ids, logged_candidates, positive_ids = payload
            positive_set = set(positive_ids)
            if len(positive_set) >= candidate_set_size:
                # Keep a deterministic subset of clicked items and one negative so
                # the task is not trivially all-positive.
                ordered_positive = sorted(
                    positive_set,
                    key=lambda x: _stable_int(seed, impression_id, "positive", x),
                )[: candidate_set_size - 1]
            else:
                ordered_positive = [x for x in logged_candidates if x in positive_set]

            negatives = [x for x in logged_candidates if x not in positive_set]
            negatives.sort(key=lambda x: _stable_int(seed, impression_id, "negative", x))
            needed = candidate_set_size - len(ordered_positive)
            selected_candidates = ordered_positive + negatives[:needed]
            if len(selected_candidates) < candidate_set_size:
                continue

            random.Random(_stable_int(seed, "candidate_order", impression_id)).shuffle(
                selected_candidates
            )
            relevant = frozenset(x for x in ordered_positive if x in selected_candidates)
            history = [news[x] for x in history_ids[-max_history_items:]]
            candidates = [news[x] for x in selected_candidates]

            instance = UserInstance(
                dataset=self.dataset_id,
                user_id=f"{user_id}:{impression_id}",
                history=history,
                candidates=candidates,
                relevant_item_ids=relevant,
                demographics={},
                instance_metadata={
                    "mind_user_id": user_id,
                    "impression_id": impression_id,
                    "impression_timestamp": timestamp,
                    "candidate_policy": "logged_impression_positive_preserving_subsample",
                    "candidate_seed": seed,
                    "identity_status": "no_observed_demographics_synthetic_stress_only",
                },
            )
            instance.validate()
            chosen_ids.append(impression_id)
            emitted += 1
            yield instance

        self._manifest = {
            "dataset_id": self.dataset_id,
            "source_version": "MIND exact split directory supplied at runtime",
            "news_loaded": len(news),
            "behavior_rows_seen": rows_seen,
            "eligible_impressions": len(eligible),
            "requested_tasks": users,
            "emitted_tasks": emitted,
            "candidate_set_size": candidate_set_size,
            "max_history_items": max_history_items,
            "seed": seed,
            "skipped_no_positive": skipped_no_positive,
            "skipped_small_candidate_set": skipped_small_candidate_set,
            "skipped_missing_news": skipped_missing_news,
            "identity_status": "no_observed_demographics_synthetic_stress_only",
            "impression_selection_hash": hashlib.sha256(
                "\n".join(chosen_ids).encode("utf-8")
            ).hexdigest(),
        }

    def preprocessing_manifest(self) -> dict[str, object]:
        return dict(self._manifest)
