from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .execute import load_and_verify_plan


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_SCHEMAS = {"faireval-run-v4", "faireval-run-v5"}
_REQUIRED_RUN_FIELDS = (
    "schema_version",
    "planned_cell_id",
    "dataset",
    "user_id",
    "condition_id",
    "condition_name",
    "prompt_template_id",
    "prompt_mode",
    "cue_id",
    "candidate_order_seed",
    "k",
    "repetition",
    "provider",
    "model_family",
    "requested_model_id",
    "request_utc",
    "temperature_requested",
    "top_p_requested",
    "max_output_tokens",
    "reasoning_or_thinking_setting",
    "reasoning_or_thinking_applied",
    "sampling_controls_applied",
    "sampling_policy",
    "output_token_parameter",
    "prompt_sha256",
    "request_sha256",
    "raw_response",
    "response_sha256",
    "initial_valid",
    "repair",
    "final_valid",
    "code_commit_sha",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_hex64(value: object, *, field: str, line_no: int) -> None:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"line {line_no}: {field} must be a lowercase SHA-256 hex digest")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: run row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("run log is empty")
    return rows


def _ranked_ids_from_json(text: str) -> list[str] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    values = payload.get("ranked_item_ids")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    return [str(value) for value in values]


def _audit_v4_repair(row: Mapping[str, Any], *, line_no: int) -> str:
    initial_valid = row["initial_valid"]
    final_valid = row["final_valid"]
    if not isinstance(initial_valid, bool) or not isinstance(final_valid, bool):
        raise ValueError(f"line {line_no}: V4 validity fields must be boolean")

    repair = row.get("repair")
    if repair is None:
        if not initial_valid and final_valid:
            raise ValueError(
                f"line {line_no}: invalid V4 initial response cannot become valid without repair"
            )
        return str(row["raw_response"])

    if initial_valid:
        raise ValueError(f"line {line_no}: V4 repair must not be present for an initially valid row")
    if not isinstance(repair, Mapping):
        raise ValueError(f"line {line_no}: V4 repair must be an object or null")
    for field in (
        "prompt_sha256",
        "raw_response",
        "response_sha256",
        "provider_metadata",
        "valid",
        "errors",
    ):
        if field not in repair:
            raise ValueError(f"line {line_no}: V4 repair missing required field {field}")
    _require_hex64(repair["prompt_sha256"], field="repair.prompt_sha256", line_no=line_no)
    _require_hex64(repair["response_sha256"], field="repair.response_sha256", line_no=line_no)
    repair_text = str(repair["raw_response"])
    if _sha256(repair_text) != repair["response_sha256"]:
        raise ValueError(f"line {line_no}: V4 repair raw response SHA-256 mismatch")
    if not isinstance(repair["provider_metadata"], Mapping):
        raise ValueError(f"line {line_no}: V4 repair provider_metadata must be an object")
    if not isinstance(repair["valid"], bool):
        raise ValueError(f"line {line_no}: V4 repair valid flag must be boolean")
    if bool(repair["valid"]) != final_valid:
        raise ValueError(f"line {line_no}: V4 repair validity disagrees with final_valid")
    return repair_text if bool(repair["valid"]) else str(row["raw_response"])


def _audit_v5_protocol(row: Mapping[str, Any], *, line_no: int) -> None:
    required = (
        "output_protocol",
        "strict_format_valid",
        "semantic_ranking_valid",
        "format_violations",
        "semantic_errors",
        "deterministic_normalization_applied",
        "normalization_actions",
        "parser_ambiguity",
        "candidate_id_mutation_detected",
        "generative_format_repair_enabled",
        "provider_generation_calls_for_cell",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"line {line_no}: V5 row missing output-protocol fields {missing}")
    if row.get("repair") is not None:
        raise ValueError(f"line {line_no}: canonical V5 must not contain generative repair")
    if row.get("generative_format_repair_enabled") is not False:
        raise ValueError(f"line {line_no}: V5 generative format repair must be false")
    if row.get("provider_generation_calls_for_cell") != 1:
        raise ValueError(f"line {line_no}: V5 requires exactly one provider generation per cell")
    if row.get("candidate_id_mutation_detected") is not False:
        raise ValueError(f"line {line_no}: V5 candidate-ID mutation is a hard failure")

    protocol = row.get("output_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(f"line {line_no}: V5 output_protocol must be an object")
    comparisons = {
        "strict_format_valid": row.get("strict_format_valid"),
        "semantic_ranking_valid": row.get("semantic_ranking_valid"),
        "format_violations": row.get("format_violations"),
        "semantic_errors": row.get("semantic_errors"),
        "normalization_applied": row.get("deterministic_normalization_applied"),
        "normalization_actions": row.get("normalization_actions"),
        "parser_ambiguity": row.get("parser_ambiguity"),
        "candidate_id_mutation_detected": row.get("candidate_id_mutation_detected"),
        "ranking": row.get("ranking"),
    }
    for key, expected in comparisons.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"line {line_no}: V5 top-level/output_protocol disagreement for {key}"
            )

    semantic_valid = row.get("semantic_ranking_valid")
    if not isinstance(semantic_valid, bool):
        raise ValueError(f"line {line_no}: V5 semantic_ranking_valid must be boolean")
    if row.get("initial_valid") is not semantic_valid or row.get("final_valid") is not semantic_valid:
        raise ValueError(f"line {line_no}: V5 legacy validity aliases must equal semantic validity")
    if row.get("parser_ambiguity") is not False:
        raise ValueError(f"line {line_no}: parser ambiguity fails canonical V5 audit")


def _check_plan_alignment(
    row: Mapping[str, Any],
    planned: Mapping[str, Any],
    *,
    line_no: int,
) -> None:
    condition = planned.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError(f"line {line_no}: referenced plan condition is malformed")

    comparisons = {
        "dataset": planned.get("dataset"),
        "user_id": planned.get("user_id"),
        "condition_id": condition.get("condition_id"),
        "condition_name": condition.get("condition_name"),
        "model_family": planned.get("model_family"),
        "requested_model_id": planned.get("model_id"),
        "prompt_template_id": planned.get("template_id"),
        "prompt_mode": planned.get("prompt_mode"),
        "cue_id": planned.get("cue_id"),
        "candidate_order_seed": planned.get("candidate_order_seed"),
        "k": planned.get("k"),
        "repetition": planned.get("repetition"),
        "temperature_requested": planned.get("temperature"),
        "top_p_requested": planned.get("top_p"),
        "max_output_tokens": planned.get("max_output_tokens"),
        "reasoning_or_thinking_setting": planned.get("reasoning_or_thinking_setting"),
        "output_token_parameter": planned.get("output_token_parameter"),
        "sampling_policy_planned": planned.get("sampling_policy"),
    }
    for run_field, expected in comparisons.items():
        if run_field == "sampling_policy_planned":
            if expected in (None, ""):
                raise ValueError(f"line {line_no}: plan lacks sampling policy")
            continue
        actual = row.get(run_field)
        if actual != expected:
            raise ValueError(
                f"line {line_no}: {run_field} disagrees with immutable plan: "
                f"run={actual!r}, plan={expected!r}"
            )

    requested_reasoning = str(planned.get("reasoning_or_thinking_setting") or "")
    applied_reasoning = str(row.get("reasoning_or_thinking_applied") or "")
    if applied_reasoning != requested_reasoning:
        raise ValueError(
            f"line {line_no}: applied reasoning/thinking mode drift: "
            f"requested={requested_reasoning!r}, applied={applied_reasoning!r}"
        )


def audit_run_log(
    output_jsonl: Path,
    *,
    plan_dir: Path | None = None,
) -> dict[str, Any]:
    """Reject mixed, tampered, provenance-incomplete, or ambiguous run logs."""
    rows = _load_rows(output_jsonl)

    plan_by_id: dict[str, dict[str, Any]] | None = None
    plan_manifest: dict[str, Any] | None = None
    if plan_dir is not None:
        plan_rows, plan_manifest = load_and_verify_plan(plan_dir)
        plan_by_id = {str(row["cell_id"]): row for row in plan_rows}

    seen_cells: set[str] = set()
    commit_shas: set[str] = set()
    model_families: set[str] = set()
    schemas: set[str] = set()
    invalid_count = 0
    strict_invalid_count = 0
    repaired_count = 0
    normalized_count = 0
    parser_ambiguity_count = 0

    for line_no, row in enumerate(rows, start=1):
        missing = [field for field in _REQUIRED_RUN_FIELDS if field not in row]
        if missing:
            raise ValueError(f"line {line_no}: missing required run fields {missing}")
        schema = str(row["schema_version"])
        if schema not in _SUPPORTED_SCHEMAS:
            raise ValueError(f"line {line_no}: unsupported schema {schema!r}")
        schemas.add(schema)

        cell_id = row["planned_cell_id"]
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(f"line {line_no}: planned_cell_id is required for frozen-run audit")
        _require_hex64(cell_id, field="planned_cell_id", line_no=line_no)
        if cell_id in seen_cells:
            raise ValueError(f"line {line_no}: duplicate planned_cell_id {cell_id}")
        seen_cells.add(cell_id)

        for field in ("prompt_sha256", "request_sha256", "response_sha256"):
            _require_hex64(row[field], field=field, line_no=line_no)
        if _sha256(str(row["raw_response"])) != row["response_sha256"]:
            raise ValueError(f"line {line_no}: raw response SHA-256 mismatch")

        if not isinstance(row["k"], int) or int(row["k"]) <= 0:
            raise ValueError(f"line {line_no}: k must be a positive integer")

        code_sha = row["code_commit_sha"]
        if not isinstance(code_sha, str) or not code_sha.strip():
            raise ValueError(f"line {line_no}: code_commit_sha is required")
        commit_shas.add(code_sha.strip())
        model_families.add(str(row["model_family"]))

        provider_metadata = row.get("provider_metadata")
        if not isinstance(provider_metadata, Mapping):
            raise ValueError(f"line {line_no}: provider_metadata must be an object")
        for field in (
            "reasoning_or_thinking_applied",
            "sampling_controls_applied",
            "sampling_policy",
            "output_token_parameter",
        ):
            if row[field] != provider_metadata.get(field):
                raise ValueError(
                    f"line {line_no}: top-level {field} disagrees with provider metadata"
                )

        if schema == "faireval-run-v4":
            authoritative_response = _audit_v4_repair(row, line_no=line_no)
            if row.get("repair") is not None:
                repaired_count += 1
            if bool(row["final_valid"]):
                ranking = row.get("ranking")
                if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
                    raise ValueError(f"line {line_no}: valid V4 row must contain a ranking list")
                values = [str(value) for value in ranking]
                if len(values) != int(row["k"]) or len(values) != len(set(values)):
                    raise ValueError(f"line {line_no}: malformed valid V4 ranking")
                if _ranked_ids_from_json(authoritative_response) != values:
                    raise ValueError(
                        f"line {line_no}: persisted V4 ranking does not match authoritative response"
                    )
            else:
                invalid_count += 1
                if row.get("ranking") is not None:
                    raise ValueError(f"line {line_no}: invalid V4 row must not persist a ranking")
        else:
            _audit_v5_protocol(row, line_no=line_no)
            if not bool(row["semantic_ranking_valid"]):
                invalid_count += 1
                if row.get("ranking") is not None:
                    raise ValueError(f"line {line_no}: invalid V5 row must not persist a ranking")
            if not bool(row["strict_format_valid"]):
                strict_invalid_count += 1
            if bool(row["deterministic_normalization_applied"]):
                normalized_count += 1
            if bool(row["parser_ambiguity"]):
                parser_ambiguity_count += 1

        if plan_by_id is not None:
            planned = plan_by_id.get(cell_id)
            if planned is None:
                raise ValueError(f"line {line_no}: cell is not present in supplied run plan")
            _check_plan_alignment(row, planned, line_no=line_no)

    if len(schemas) != 1:
        raise ValueError(f"run log mixes schemas: {sorted(schemas)}")
    if len(commit_shas) != 1:
        raise ValueError(
            "run log mixes multiple code commits; split or explicitly version the experiment: "
            f"{sorted(commit_shas)}"
        )

    return {
        "schema_version": "faireval-run-audit-v5",
        "run_schema_version": next(iter(schemas)),
        "rows": len(rows),
        "unique_planned_cells": len(seen_cells),
        "semantic_invalid_outputs": invalid_count,
        "strict_format_invalid_outputs": strict_invalid_count,
        "rows_with_legacy_generative_repair": repaired_count,
        "rows_with_deterministic_normalization": normalized_count,
        "parser_ambiguity_rows": parser_ambiguity_count,
        "model_families": sorted(model_families),
        "code_commit_sha": next(iter(commit_shas)),
        "plan_sha256": None if plan_manifest is None else plan_manifest.get("plan_sha256"),
        "status": "pass",
    }
