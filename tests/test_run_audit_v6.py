import hashlib
import json
from pathlib import Path

from faireval.output_protocol import analyze_ranking_output
from faireval.run_audit import audit_run_log


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row() -> dict:
    raw = '{"ranked_item_ids":["i001","i002"]}'
    protocol = analyze_ranking_output(
        raw,
        candidate_ids=("i001", "i002", "i003"),
        k=2,
    ).as_dict()
    metadata = {
        "reasoning_or_thinking_applied": "not_applicable",
        "sampling_controls_applied": True,
        "sampling_policy": "explicit_temperature_and_top_p",
        "output_token_parameter": "max_new_tokens",
    }
    return {
        "schema_version": "faireval-run-v6",
        "prompt_interface_version": "faireval-prompt-interface-v6",
        "planned_cell_id": "a" * 64,
        "dataset": "fairsynth360",
        "user_id": "s0001",
        "condition_id": "C0",
        "condition_name": "preference_only",
        "template_id": "field_v2_a",
        "prompt_template_id": "field_v2_a",
        "prompt_mode": "audit",
        "cue_id": "structured_key_value",
        "candidate_order_seed": 1729,
        "k": 2,
        "repetition": 0,
        "provider": "local_transformers",
        "model_family": "qwen25_local",
        "requested_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "resolved_model_version": "a09a35458c702b33eeacc393d103063234e8bc28",
        "request_utc": "2026-09-23T00:00:00+00:00",
        "temperature": 0.2,
        "temperature_requested": 0.2,
        "top_p": 1.0,
        "top_p_requested": 1.0,
        "max_output_tokens": 512,
        "reasoning_or_thinking_setting": "not_applicable",
        "reasoning_or_thinking_applied": "not_applicable",
        "sampling_controls_applied": True,
        "sampling_policy": "explicit_temperature_and_top_p",
        "output_token_parameter": "max_new_tokens",
        "seed_requested": 7,
        "seed_supported": True,
        "prompt_sha256": "b" * 64,
        "request_sha256": "c" * 64,
        "raw_response": raw,
        "response_sha256": _sha(raw),
        "provider_metadata": metadata,
        "initial_valid": True,
        "initial_errors": [],
        "repair": None,
        "final_valid": True,
        "final_errors": [],
        "ranking": ["i001", "i002"],
        "output_protocol": protocol,
        "strict_format_valid": True,
        "semantic_ranking_valid": True,
        "format_violations": [],
        "semantic_errors": [],
        "deterministic_normalization_applied": False,
        "normalization_actions": [],
        "parser_ambiguity": False,
        "candidate_id_mutation_detected": False,
        "generative_format_repair_enabled": False,
        "provider_generation_calls_for_cell": 1,
        "code_commit_sha": "deadbeef",
    }


def test_v6_run_audit_accepts_versioned_prompt_interface(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(_row(), sort_keys=True) + "\n", encoding="utf-8")
    result = audit_run_log(path)
    assert result["status"] == "pass"
    assert result["run_schema_version"] == "faireval-run-v6"
    assert result["schema_version"] == "faireval-run-audit-v6"


def test_v6_run_audit_rejects_missing_prompt_interface_version(tmp_path: Path):
    row = _row()
    row["prompt_interface_version"] = "faireval-prompt-interface-v5"
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    try:
        audit_run_log(path)
    except ValueError as exc:
        assert "faireval-prompt-interface-v6" in str(exc)
    else:
        raise AssertionError("V6 audit should reject prompt-interface drift")
