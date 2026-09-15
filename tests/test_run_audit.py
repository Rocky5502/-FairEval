import hashlib
import json
from pathlib import Path

import pytest

from faireval.run_audit import audit_run_log


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row() -> dict:
    raw = '{"ranked_item_ids":["a","b"]}'
    return {
        "schema_version": "faireval-run-v4",
        "planned_cell_id": "a" * 64,
        "dataset": "movielens_1m",
        "user_id": "u1",
        "condition_id": "C0",
        "condition_name": "preference_only",
        "template_id": "field_v2_a",
        "prompt_template_id": "field_v2_a",
        "prompt_mode": "audit",
        "cue_id": "structured_key_value",
        "candidate_order_seed": None,
        "k": 2,
        "repetition": 0,
        "provider": "openai",
        "model_family": "openai",
        "requested_model_id": "gpt-5.6-terra",
        "resolved_model_version": "gpt-5.6-terra",
        "request_utc": "2026-09-15T00:00:00+00:00",
        "temperature": 0.2,
        "temperature_requested": 0.2,
        "top_p": 1.0,
        "top_p_requested": 1.0,
        "max_output_tokens": 512,
        "reasoning_or_thinking_setting": "none",
        "reasoning_or_thinking_applied": "none",
        "sampling_controls_applied": True,
        "sampling_policy": "explicit_temperature_and_top_p",
        "output_token_parameter": "max_completion_tokens",
        "seed_requested": None,
        "seed_supported": False,
        "prompt_sha256": "b" * 64,
        "request_sha256": "c" * 64,
        "raw_response": raw,
        "response_sha256": _sha(raw),
        "provider_metadata": {
            "sampling_controls_applied": True,
            "sampling_policy": "explicit_temperature_and_top_p",
            "output_token_parameter": "max_completion_tokens",
        },
        "initial_valid": True,
        "initial_errors": [],
        "repair": None,
        "final_valid": True,
        "final_errors": [],
        "ranking": ["a", "b"],
        "code_commit_sha": "deadbeef",
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_run_audit_accepts_complete_provenance(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    _write(path, [_row()])
    result = audit_run_log(path)
    assert result["status"] == "pass"
    assert result["rows"] == 1
    assert result["invalid_outputs"] == 0


def test_run_audit_rejects_tampered_raw_response(tmp_path: Path):
    row = _row()
    row["raw_response"] = "tampered"
    path = tmp_path / "runs.jsonl"
    _write(path, [row])
    with pytest.raises(ValueError, match="response SHA-256 mismatch"):
        audit_run_log(path)


def test_run_audit_rejects_duplicate_planned_cell(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    _write(path, [_row(), _row()])
    with pytest.raises(ValueError, match="duplicate planned_cell_id"):
        audit_run_log(path)


def test_run_audit_rejects_mixed_code_commits(tmp_path: Path):
    first = _row()
    second = _row()
    second["planned_cell_id"] = "d" * 64
    second["code_commit_sha"] = "cafebabe"
    path = tmp_path / "runs.jsonl"
    _write(path, [first, second])
    with pytest.raises(ValueError, match="mixes multiple code commits"):
        audit_run_log(path)


def test_run_audit_rejects_provider_metadata_drift(tmp_path: Path):
    row = _row()
    row["provider_metadata"]["output_token_parameter"] = "max_tokens"
    path = tmp_path / "runs.jsonl"
    _write(path, [row])
    with pytest.raises(ValueError, match="output_token_parameter disagrees"):
        audit_run_log(path)


def test_run_audit_rejects_ranking_length_that_disagrees_with_k(tmp_path: Path):
    row = _row()
    row["k"] = 3
    path = tmp_path / "runs.jsonl"
    _write(path, [row])
    with pytest.raises(ValueError, match="does not equal k"):
        audit_run_log(path)
