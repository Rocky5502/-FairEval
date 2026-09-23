from __future__ import annotations

from .output_protocol import analyze_ranking_output
from .schema import RankingOutput, UserInstance, ValidationResult


def validate_ranking_output(
    raw_text: str,
    instance: UserInstance,
    *,
    k: int,
    repaired_format: bool = False,
) -> ValidationResult:
    """Compatibility validator backed by the deterministic V5 output protocol.

    valid means the response contains an exact-k, unique, candidate-only ranking
    after envelope-only deterministic normalization. Formatting violations remain
    visible through analyze_ranking_output and are not silently treated as clean
    strict-format compliance.
    """
    result = analyze_ranking_output(
        raw_text,
        candidate_ids=instance.candidate_ids(),
        k=k,
    )
    if not result.semantic_ranking_valid or result.ranking is None:
        errors = tuple(sorted(set(result.format_violations + result.semantic_errors)))
        return ValidationResult(valid=False, errors=errors)

    return ValidationResult(
        valid=True,
        errors=tuple(result.format_violations),
        ranking=RankingOutput(
            ranked_item_ids=result.ranking,
            raw_text=raw_text,
            repaired_format=bool(repaired_format or result.normalization_applied),
        ),
    )


def build_format_repair_prompt(raw_text: str, *, k: int) -> str:
    """Legacy V4 helper retained only for forensic reproducibility.

    Canonical V5 execution MUST NOT call a model to repair output. The V5 runner
    rejects generative repair and uses deterministic envelope normalization only.
    """
    return (
        "Reformat the following recommendation into JSON only. Preserve the exact item choices "
        "and their order; do not add, remove, replace, or rerank any item. "
        f'Return exactly this schema with {k} IDs: {{"ranked_item_ids":[...]}}.\n\n'
        f"ORIGINAL_OUTPUT:\n{raw_text}"
    )
