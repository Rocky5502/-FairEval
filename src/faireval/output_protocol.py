from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


PROTOCOL_VERSION = "faireval-output-protocol-v5"

_FENCE_RE = re.compile(
    r"\A```(?P<lang>json)?[ \t]*\r?\n(?P<body>.*)\r?\n```[ \t]*\Z",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class OutputProtocolResult:
    protocol_version: str
    strict_format_valid: bool
    semantic_ranking_valid: bool
    format_violations: tuple[str, ...]
    semantic_errors: tuple[str, ...]
    ranking: tuple[str, ...] | None
    normalization_applied: bool
    normalization_actions: tuple[str, ...]
    parser_ambiguity: bool
    candidate_id_mutation_detected: bool
    top_level_type: str | None
    forensic_class: str
    normalized_envelope_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["format_violations"] = list(self.format_violations)
        payload["semantic_errors"] = list(self.semantic_errors)
        payload["ranking"] = None if self.ranking is None else list(self.ranking)
        payload["normalization_actions"] = list(self.normalization_actions)
        return payload


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _single_embedded_object(text: str) -> tuple[str | None, bool]:
    """Return one mechanically unambiguous embedded JSON object, if present.

    JSON tokens are never rewritten. The standard decoder only identifies an
    exact object substring. Multiple distinct objects are parser ambiguity.
    """
    decoder = json.JSONDecoder()
    matches: list[tuple[int, int, str]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        end = index + consumed
        key = (index, end, text[index:end])
        if key not in matches:
            matches.append(key)
    if len(matches) == 1:
        return matches[0][2], False
    return None, len(matches) > 1


def _primary_class(
    *,
    strict_format_valid: bool,
    semantic_ranking_valid: bool,
    normalization_applied: bool,
    format_violations: Iterable[str],
    semantic_errors: Iterable[str],
) -> str:
    format_set = set(format_violations)
    semantic_set = set(semantic_errors)
    if strict_format_valid and semantic_ranking_valid:
        return "A_strict_exact_json_valid"
    if semantic_ranking_valid and normalization_applied:
        return "B_deterministic_envelope_recoverable"
    if (
        "wrong_top_level_schema" in semantic_set
        or "missing_ranked_item_ids" in semantic_set
        or "nonlist_ranked_item_ids" in semantic_set
        or "extra_keys" in format_set
    ):
        return "C_syntactically_parseable_wrong_schema"
    if "wrong_k" in semantic_set:
        return "D_wrong_k"
    if "duplicate_item_ids" in semantic_set:
        return "E_duplicate"
    if "out_of_candidate_item" in semantic_set:
        return "F_out_of_candidate"
    return "G_genuinely_unrecoverable"


def analyze_ranking_output(
    raw_text: str,
    *,
    candidate_ids: Iterable[str],
    k: int,
) -> OutputProtocolResult:
    """Deterministically classify one recommendation response.

    Normalization is envelope-only: surrounding whitespace, one recognized
    Markdown JSON fence, or one unambiguous prose-wrapped JSON object may be
    removed. Item IDs, rank order, list length, and JSON values are never changed.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    raw = str(raw_text)
    candidate_set = {str(value) for value in candidate_ids}
    format_violations: list[str] = []
    semantic_errors: list[str] = []
    normalization_actions: list[str] = []
    parser_ambiguity = False

    trimmed = raw.strip()
    if trimmed != raw:
        format_violations.append("surrounding_whitespace")
        normalization_actions.append("strip_surrounding_whitespace")

    candidate_text = trimmed
    parsed_directly = False
    try:
        parsed: Any = json.loads(candidate_text)
        parsed_directly = True
    except json.JSONDecodeError:
        parsed = None

    if not parsed_directly:
        fence = _FENCE_RE.fullmatch(trimmed)
        if fence is not None:
            format_violations.append("markdown_fence")
            normalization_actions.append("strip_single_recognized_json_fence")
            candidate_text = fence.group("body").strip()
            try:
                parsed = json.loads(candidate_text)
            except json.JSONDecodeError:
                parsed = None
                semantic_errors.append("invalid_json_syntax")
        else:
            embedded, ambiguous = _single_embedded_object(trimmed)
            parser_ambiguity = ambiguous
            if ambiguous:
                semantic_errors.append("parser_ambiguity")
                parsed = None
            elif embedded is not None:
                format_violations.append("extra_text")
                normalization_actions.append("extract_single_unambiguous_json_object")
                candidate_text = embedded
                try:
                    parsed = json.loads(candidate_text)
                except json.JSONDecodeError:
                    parsed = None
                    semantic_errors.append("invalid_json_syntax")
            else:
                semantic_errors.append("invalid_json_syntax")
                parsed = None

    top_level_type = None if parsed is None else type(parsed).__name__
    strict_schema_shape = False
    ranking: tuple[str, ...] | None = None

    if parsed is not None:
        if not isinstance(parsed, dict):
            semantic_errors.append("wrong_top_level_schema")
        else:
            keys = set(parsed)
            extra_keys = sorted(keys - {"ranked_item_ids"})
            if extra_keys:
                format_violations.append("extra_keys")
            if "ranked_item_ids" not in parsed:
                semantic_errors.append("missing_ranked_item_ids")
            else:
                ranked = parsed["ranked_item_ids"]
                if not isinstance(ranked, list):
                    semantic_errors.append("nonlist_ranked_item_ids")
                else:
                    strict_schema_shape = not extra_keys
                    if len(ranked) != k:
                        semantic_errors.append("wrong_k")

                    strings: list[str] = []
                    for value in ranked:
                        if not isinstance(value, str):
                            semantic_errors.append("non_string_item_id")
                            if isinstance(value, (list, dict, tuple, set)):
                                semantic_errors.append("non_scalar_item_id")
                            continue
                        strings.append(value)

                    if len(strings) == len(ranked):
                        if len(set(strings)) != len(strings):
                            semantic_errors.append("duplicate_item_ids")
                        if any(value not in candidate_set for value in strings):
                            semantic_errors.append("out_of_candidate_item")

                    if not semantic_errors:
                        ranking = tuple(strings)

    strict_format_valid = (
        parsed_directly
        and raw == trimmed
        and isinstance(parsed, dict)
        and strict_schema_shape
        and isinstance(parsed.get("ranked_item_ids"), list)
        and not format_violations
    )
    semantic_ranking_valid = ranking is not None
    normalization_applied = bool(normalization_actions)

    # Envelope normalization never edits parsed ranking tokens. Any future
    # normalizer that rewrites IDs must change this invariant and fail the gate.
    candidate_id_mutation_detected = False

    format_tuple = tuple(sorted(set(format_violations)))
    semantic_tuple = tuple(sorted(set(semantic_errors)))
    forensic_class = _primary_class(
        strict_format_valid=strict_format_valid,
        semantic_ranking_valid=semantic_ranking_valid,
        normalization_applied=normalization_applied,
        format_violations=format_tuple,
        semantic_errors=semantic_tuple,
    )
    return OutputProtocolResult(
        protocol_version=PROTOCOL_VERSION,
        strict_format_valid=strict_format_valid,
        semantic_ranking_valid=semantic_ranking_valid,
        format_violations=format_tuple,
        semantic_errors=semantic_tuple,
        ranking=ranking,
        normalization_applied=normalization_applied,
        normalization_actions=tuple(normalization_actions),
        parser_ambiguity=parser_ambiguity,
        candidate_id_mutation_detected=candidate_id_mutation_detected,
        top_level_type=top_level_type,
        forensic_class=forensic_class,
        normalized_envelope_sha256=None if parsed is None else _sha256(candidate_text),
    )
