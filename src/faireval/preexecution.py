from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .execute import load_and_verify_plan


SEAL_SCHEMA = "faireval-preexecution-seal-v1"


def load_preexecution_seal(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"pre-execution seal not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pre-execution seal must contain a JSON object")
    if payload.get("schema_version") != SEAL_SCHEMA:
        raise ValueError(
            f"unsupported pre-execution seal schema {payload.get('schema_version')!r}"
        )
    return payload


def verify_preexecution_seal(
    path: Path,
    *,
    expected_commit_sha: str,
    plan_dir: Path,
    plan_key: str,
) -> dict[str, Any]:
    """Verify that execution uses the exact sealed commit and deterministic plan."""
    seal = load_preexecution_seal(path)
    sealed_commit = str(seal.get("git_commit_sha", ""))
    if sealed_commit != expected_commit_sha:
        raise ValueError(
            "pre-execution seal commit mismatch: "
            f"seal={sealed_commit!r}, execution={expected_commit_sha!r}"
        )

    if seal.get("hosted_api_generation_calls_made") != 0:
        raise ValueError("pre-execution seal is contaminated by hosted generation calls")
    if seal.get("local_model_weights_loaded") is not False:
        raise ValueError("pre-execution seal is contaminated by local model loading")
    if seal.get("empirical_results_seen_or_inserted") is not False:
        raise ValueError("pre-execution seal is contaminated by empirical results")

    plans = seal.get("plans")
    if not isinstance(plans, dict) or plan_key not in plans:
        raise ValueError(f"pre-execution seal lacks required plan block {plan_key!r}")
    sealed_plan = plans[plan_key]
    if not isinstance(sealed_plan, dict):
        raise ValueError(f"pre-execution seal plan block {plan_key!r} is malformed")

    rows, manifest = load_and_verify_plan(plan_dir)
    actual_plan_sha = manifest.get("plan_sha256")
    sealed_plan_sha = sealed_plan.get("plan_sha256")
    if actual_plan_sha != sealed_plan_sha:
        raise ValueError(
            "pre-execution seal plan mismatch: "
            f"seal={sealed_plan_sha!r}, actual={actual_plan_sha!r}"
        )
    sealed_cells = int(sealed_plan.get("planned_cells", -1))
    if sealed_cells != len(rows):
        raise ValueError(
            "pre-execution seal planned-cell count mismatch: "
            f"seal={sealed_cells}, actual={len(rows)}"
        )

    return {
        "schema_version": "faireval-preexecution-seal-verification-v1",
        "status": "pass",
        "seal_path": str(path),
        "git_commit_sha": sealed_commit,
        "scientific_spec_sha256": seal.get("scientific_spec_sha256"),
        "plan_key": plan_key,
        "plan_sha256": actual_plan_sha,
        "planned_cells": len(rows),
    }
