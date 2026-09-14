from __future__ import annotations

import json
from typing import Any

from .schema import RankingOutput, UserInstance, ValidationResult


def _extract_payload(raw_text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def validate_ranking_output(
    raw_text: str,
    instance: UserInstance,
    *,
    k: int,
    repaired_format: bool = False,
) -> ValidationResult:
    """Validate one model output without silently deleting failures."""
    payload = _extract_payload(raw_text)
    if payload is None:
        return ValidationResult(valid=False, errors=("invalid_json",))

    ranked = payload.get("ranked_item_ids")
    if not isinstance(ranked, list):
        return ValidationResult(valid=False, errors=("missing_or_nonlist_ranked_item_ids",))

    errors: list[str] = []
    normalized: list[str] = []
    for value in ranked:
        if not isinstance(value, (str, int)):
            errors.append("non_scalar_item_id")
            continue
        normalized.append(str(value))

    if len(normalized) != k:
        errors.append("wrong_k")
    if len(set(normalized)) != len(normalized):
        errors.append("duplicate_item_ids")

    candidate_ids = set(instance.candidate_ids())
    if any(item_id not in candidate_ids for item_id in normalized):
        errors.append("out_of_candidate_item")

    if errors:
        return ValidationResult(valid=False, errors=tuple(sorted(set(errors))))

    return ValidationResult(
        valid=True,
        errors=(),
        ranking=RankingOutput(
            ranked_item_ids=tuple(normalized),
            raw_text=raw_text,
            repaired_format=repaired_format,
        ),
    )


def build_format_repair_prompt(raw_text: str, *, k: int) -> str:
    """One format-only repair instruction.

    The repair model is not allowed to add/remove/reorder recommendation choices;
    it may only serialize already-present candidate IDs into the required schema.
    """
    return (
        "Reformat the following recommendation into JSON only. Preserve the exact item choices "
        "and their order; do not add, remove, replace, or rerank any item. "
        f'Return exactly this schema with {k} IDs: {{"ranked_item_ids":[...]}}.\n\n'
        f"ORIGINAL_OUTPUT:\n{raw_text}"
    )
