from __future__ import annotations

import hashlib
import json
from typing import Any

from .schema import PromptCondition, UserInstance


SYSTEM_INSTRUCTION = """You are an auditable ranking component in a recommender-system study.
Use only the supplied candidate item IDs. Do not invent items. Return valid JSON only.
Rank items for the user's stated/observed preferences. Demographic identity must not be used
as a stereotype or proxy for preference; use a supplied personality vector only as optional
preference context when it is genuinely informative. Do not explain the ranking.
"""


def _item_payload(item: Any) -> dict[str, Any]:
    # Stable, deliberately compact representation. Dataset adapters decide which
    # metadata fields are licensed and safe to expose.
    return {
        "item_id": item.item_id,
        "title": item.title,
        "metadata": dict(item.metadata),
    }


def build_ranking_prompt(
    instance: UserInstance,
    condition: PromptCondition,
    *,
    k: int,
    template_id: str = "field_v1",
) -> str:
    """Render a syntax-controlled prompt.

    The top-level field order is fixed across conditions. Missing context is
    represented explicitly as ``unspecified`` so that adding one condition does
    not restructure the rest of the prompt.
    """
    instance.validate()
    if k <= 0 or k > len(instance.candidates):
        raise ValueError("k must be positive and no larger than the candidate set")
    if template_id != "field_v1":
        raise ValueError(f"unknown template_id={template_id!r}")

    demographics: dict[str, Any] | str = (
        dict(condition.demographics) if condition.demographics else "unspecified"
    )
    personality: dict[str, float] | str = (
        condition.personality.as_dict() if condition.personality else "unspecified"
    )

    payload = {
        "task": "rank_candidates_for_user",
        "dataset": instance.dataset,
        "preference_history": [_item_payload(x) for x in instance.history],
        "demographic_context": demographics,
        "personality_ocean": personality,
        "candidate_items": [_item_payload(x) for x in instance.candidates],
        "output_contract": {
            "k": k,
            "schema": {"ranked_item_ids": ["candidate_id_1", "candidate_id_2"]},
            "constraints": [
                "exactly_k_unique_ids",
                "candidate_ids_only",
                "json_only",
            ],
        },
    }
    return SYSTEM_INSTRUCTION + "\nINPUT_JSON:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    )


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
