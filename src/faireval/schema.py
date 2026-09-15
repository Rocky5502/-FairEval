from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


OCEAN_KEYS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


@dataclass(frozen=True)
class Item:
    """A candidate item shown to an LLM recommender."""

    item_id: str
    title: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PersonalityProfile:
    """Standardized Big Five/OCEAN values.

    Values are expected to be scaled to [0, 1] by a dataset-specific adapter.
    The original raw instrument scores must remain available in dataset cards
    or preprocessing artifacts; this class stores the normalized prompt view.
    """

    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def as_dict(self) -> dict[str, float]:
        values = {key: float(getattr(self, key)) for key in OCEAN_KEYS}
        for key, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be in [0,1], got {value}")
        return values


@dataclass(frozen=True)
class UserInstance:
    """One auditable recommendation task shared by every experimental condition."""

    dataset: str
    user_id: str
    history: Sequence[Item]
    candidates: Sequence[Item]
    relevant_item_ids: frozenset[str]
    demographics: Mapping[str, Any] = field(default_factory=dict)
    personality: PersonalityProfile | None = None
    instance_metadata: Mapping[str, Any] = field(default_factory=dict)

    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.candidates)

    def validate(self) -> None:
        ids = self.candidate_ids()
        if not ids:
            raise ValueError("candidate set cannot be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")
        if not self.relevant_item_ids.issubset(set(ids)):
            raise ValueError("all held-out relevant items must be in the candidate set")


@dataclass(frozen=True)
class PromptCondition:
    condition_id: str
    condition_name: str
    demographics: Mapping[str, Any] | None = None
    personality: PersonalityProfile | None = None
    intervention: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingOutput:
    ranked_item_ids: tuple[str, ...]
    raw_text: str = ""
    repaired_format: bool = False


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]
    ranking: RankingOutput | None = None


@dataclass(frozen=True)
class RunMetadata:
    provider: str
    model_family: str
    requested_model_id: str
    resolved_model_version: str | None
    request_utc: str
    repetition: int
    temperature: float | None
    top_p: float | None
    max_output_tokens: int | None
    reasoning_or_thinking_setting: str | None
    seed_requested: int | None
    seed_supported: bool | None
    prompt_sha256: str
    request_sha256: str | None = None
    response_sha256: str | None = None
    code_commit_sha: str | None = None
