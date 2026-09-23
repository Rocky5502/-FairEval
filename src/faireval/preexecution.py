from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .execute import load_and_verify_plan


SEAL_SCHEMAS = {
    "faireval-preexecution-seal-v1",
    "faireval-preexecution-seal-v2",
    "faireval-preexecution-seal-v3",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_preexecution_seal(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"pre-execution seal not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pre-execution seal must contain a JSON object")
    if payload.get("schema_version") not in SEAL_SCHEMAS:
        raise ValueError(
            f"unsupported pre-execution seal schema {payload.get('schema_version')!r}"
        )
    return payload


def _verify_scientific_spec_files(
    seal: dict[str, Any],
    *,
    spec_root: Path,
) -> tuple[str, int]:
    sealed_files = seal.get("scientific_spec_files")
    if not isinstance(sealed_files, dict) or not sealed_files:
        raise ValueError("pre-execution seal lacks scientific_spec_files")

    actual: dict[str, str] = {}
    for raw_name, raw_digest in sorted(sealed_files.items()):
        name = str(raw_name).replace("\\", "/")
        expected_digest = str(raw_digest).lower()
        if len(expected_digest) != 64:
            raise ValueError(f"pre-execution seal has invalid file digest for {name!r}")
        path = spec_root / name
        if not path.is_file():
            raise ValueError(f"sealed scientific source is missing at execution: {name}")
        digest = _sha256(path)
        if digest != expected_digest:
            raise ValueError(
                "sealed scientific source hash mismatch: "
                f"path={name!r}, seal={expected_digest}, actual={digest}"
            )
        actual[name] = digest

    sealed_spec_digest = str(seal.get("scientific_spec_sha256", ""))
    actual_spec_digest = _json_digest(actual)
    if actual_spec_digest != sealed_spec_digest:
        raise ValueError(
            "pre-execution scientific specification digest mismatch: "
            f"seal={sealed_spec_digest!r}, actual={actual_spec_digest!r}"
        )

    sealed_count = int(seal.get("scientific_spec_file_count", -1))
    if sealed_count != len(actual):
        raise ValueError(
            "pre-execution scientific specification file-count mismatch: "
            f"seal={sealed_count}, actual={len(actual)}"
        )
    return actual_spec_digest, len(actual)


def verify_preexecution_seal(
    path: Path,
    *,
    expected_commit_sha: str,
    plan_dir: Path,
    plan_key: str,
    spec_root: Path | None = None,
) -> dict[str, Any]:
    """Verify exact sealed commit, scientific sources, and deterministic plan."""
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
    if seal.get("schema_version") == "faireval-preexecution-seal-v1":
        if seal.get("empirical_results_seen_or_inserted") is not False:
            raise ValueError("V1 pre-execution seal is contaminated by empirical results")
    elif seal.get("schema_version") == "faireval-preexecution-seal-v2":
        # V5 is a transparently versioned recovery after the frozen V4 interface
        # failure, so prior V4 outcomes are necessarily known. The scientific
        # safeguard is that no V5 canary outcome was seen before this new gate.
        if seal.get("prior_v4_results_known") is not True:
            raise ValueError("V2 seal must disclose that frozen V4 results were already known")
        if seal.get("prior_v4_protocol_failure_known") is not True:
            raise ValueError("V2 seal must disclose the known V4 protocol failure")
        if seal.get("v5_canary_results_seen_before_seal") is not False:
            raise ValueError("V2 seal must be created before any V5 canary result is inspected")
    else:
        # V6 follows a failed, fully audited V5 canary. V5 outcomes are known, so
        # the next safeguard is a new, disjoint canary whose outcomes are unseen
        # when the V6 gate is sealed.
        required_true = (
            "prior_v4_results_known",
            "prior_v4_protocol_failure_known",
            "prior_v5_canary_results_known",
            "prior_v5_canary_failed",
            "v6_canary_disjoint_from_v5_users",
        )
        for field in required_true:
            if seal.get(field) is not True:
                raise ValueError(f"V3 seal must disclose/confirm {field}")
        if seal.get("v6_canary_results_seen_before_seal") is not False:
            raise ValueError("V3 seal must be created before any V6 canary result is inspected")

    actual_spec_digest, spec_file_count = _verify_scientific_spec_files(
        seal,
        spec_root=REPO_ROOT if spec_root is None else Path(spec_root),
    )

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
        "schema_version": (
            "faireval-preexecution-seal-verification-v3"
            if seal.get("schema_version") == "faireval-preexecution-seal-v3"
            else "faireval-preexecution-seal-verification-v2"
        ),
        "status": "pass",
        "seal_path": str(path),
        "git_commit_sha": sealed_commit,
        "scientific_spec_sha256": actual_spec_digest,
        "scientific_spec_file_count": spec_file_count,
        "plan_key": plan_key,
        "plan_sha256": actual_plan_sha,
        "planned_cells": len(rows),
    }
