from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from faireval.freeze import canonical_json, file_sha256
from faireval.preexecution import verify_preexecution_seal


def _write_plan(tmp_path: Path) -> tuple[Path, str]:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    payload = {
        "schema_version": "faireval-run-plan-v1",
        "dataset": "fairsynth360",
        "user_id": "u1",
        "condition": {
            "condition_id": "C0",
            "condition_name": "preference_only",
            "demographics": None,
            "personality": None,
            "intervention": {},
        },
        "analysis_roles": ["sanity"],
        "confirmatory": False,
        "model_family": "qwen25_local",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "reasoning_or_thinking_setting": "none",
        "sampling_policy": "explicit_temperature_and_top_p",
        "output_token_parameter": "max_new_tokens",
        "template_id": "field_v2_a",
        "prompt_mode": "audit",
        "cue_id": "structured_key_value",
        "candidate_order_seed": 1,
        "k": 10,
        "repetition": 0,
        "temperature": 0.2,
        "top_p": 1.0,
        "max_output_tokens": 512,
        "seed": 1729,
    }
    cell_id = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    row = {**payload, "cell_id": cell_id}
    plan_path = plan_dir / "run_plan.jsonl"
    plan_path.write_text(canonical_json(row) + "\n", encoding="utf-8")
    plan_sha = hashlib.sha256(canonical_json([row]).encode("utf-8")).hexdigest()
    (plan_dir / "plan_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "test-plan-v1",
                "planned_api_cells": 1,
                "plan_sha256": plan_sha,
                "run_plan_file_sha256": file_sha256(plan_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_dir, plan_sha


def _write_seal(tmp_path: Path, *, commit: str, plan_sha: str) -> Path:
    path = tmp_path / "seal.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "faireval-preexecution-seal-v1",
                "git_commit_sha": commit,
                "scientific_spec_sha256": "a" * 64,
                "hosted_api_generation_calls_made": 0,
                "local_model_weights_loaded": False,
                "empirical_results_seen_or_inserted": False,
                "plans": {
                    "whitebox_core": {
                        "plan_sha256": plan_sha,
                        "planned_cells": 1,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_verify_preexecution_seal_accepts_matching_commit_and_plan(tmp_path: Path) -> None:
    plan_dir, plan_sha = _write_plan(tmp_path)
    seal = _write_seal(tmp_path, commit="abc123", plan_sha=plan_sha)
    result = verify_preexecution_seal(
        seal,
        expected_commit_sha="abc123",
        plan_dir=plan_dir,
        plan_key="whitebox_core",
    )
    assert result["status"] == "pass"
    assert result["planned_cells"] == 1


def test_verify_preexecution_seal_rejects_commit_drift(tmp_path: Path) -> None:
    plan_dir, plan_sha = _write_plan(tmp_path)
    seal = _write_seal(tmp_path, commit="sealed", plan_sha=plan_sha)
    with pytest.raises(ValueError, match="commit mismatch"):
        verify_preexecution_seal(
            seal,
            expected_commit_sha="different",
            plan_dir=plan_dir,
            plan_key="whitebox_core",
        )
