from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from faireval.datasets.factory import DATASET_IDS


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_API_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "LLAMA_PROVIDER_API_KEY",
)
REQUIRED_ENDPOINT_ENV = (
    "QWEN_BASE_URL",
    "LLAMA_BASE_URL",
    "LLAMA_PROVIDER_NAME",
)


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _dataset_blockers(repo_root: Path, lock: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    rows = lock.get("datasets")
    if not isinstance(rows, dict):
        return ["configs/dataset_releases.yaml lacks datasets mapping"]

    if set(rows) != set(DATASET_IDS):
        blockers.append(
            "dataset release lock IDs do not exactly match executable adapters: "
            f"lock={sorted(rows)} adapters={sorted(DATASET_IDS)}"
        )

    for dataset_id in DATASET_IDS:
        row = rows.get(dataset_id)
        if not isinstance(row, dict):
            blockers.append(f"{dataset_id}: release-lock row missing")
            continue
        if row.get("release_status") != "frozen":
            blockers.append(f"{dataset_id}: release_status is not frozen")
        digest = str(row.get("raw_sha256") or "").lower()
        if not SHA256_RE.fullmatch(digest):
            blockers.append(f"{dataset_id}: raw_sha256 is not a frozen SHA-256 digest")
        if row.get("license_reviewed") is not True:
            blockers.append(f"{dataset_id}: license_reviewed is not true")
        raw_path = str(row.get("raw_path") or "").strip()
        if not raw_path:
            blockers.append(f"{dataset_id}: raw_path missing")
        elif not (repo_root / raw_path).exists():
            blockers.append(f"{dataset_id}: local raw_path does not exist: {raw_path}")
        if not str(row.get("release_id") or "").strip():
            blockers.append(f"{dataset_id}: release_id missing")
    return blockers


def _model_blockers(models: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    rows = [row for row in models.get("models", []) if isinstance(row, dict) and row.get("enabled", True)]
    if len(rows) != 6:
        blockers.append(f"model panel must contain six enabled families; found {len(rows)}")
        return blockers
    by_family = {str(row.get("family")): row for row in rows}
    if set(by_family) != {"openai", "anthropic", "google", "deepseek", "qwen", "meta"}:
        blockers.append(f"unexpected enabled model families: {sorted(by_family)}")

    meta = by_family.get("meta", {})
    if not meta.get("provider"):
        blockers.append("meta: configs/models.yaml provider is not frozen")
    for family, row in by_family.items():
        for key in (
            "model_id",
            "reasoning_or_thinking_setting",
            "sampling_policy",
            "output_token_parameter",
        ):
            if not str(row.get(key) or "").strip():
                blockers.append(f"{family}: {key} missing")
    return blockers


def _environment_blockers() -> list[str]:
    blockers: list[str] = []
    for name in REQUIRED_API_ENV:
        if not os.environ.get(name):
            blockers.append(f"environment: {name} missing")
    for name in REQUIRED_ENDPOINT_ENV:
        if not os.environ.get(name):
            blockers.append(f"environment: {name} missing")
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate whether FairEval is safe to promote from repo-ready to API-pilot-ready."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero until dataset locks, provider freeze, endpoints and credentials are complete.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_lock_path = repo_root / "configs" / "dataset_releases.yaml"
    models_path = repo_root / "configs" / "models.yaml"

    dataset_lock = _load(dataset_lock_path)
    models = _load(models_path)
    blockers = _dataset_blockers(repo_root, dataset_lock) + _model_blockers(models)
    environment_blockers = _environment_blockers()

    structural_errors = [
        message
        for message in blockers
        if "IDs do not exactly match" in message
        or "release-lock row missing" in message
        or "model panel must contain" in message
        or "unexpected enabled model families" in message
        or message.endswith("missing")
    ]

    report = {
        "schema_version": "faireval-pilot-readiness-v1",
        "dataset_ids": list(DATASET_IDS),
        "dataset_and_model_blockers": blockers,
        "environment_blockers": environment_blockers,
        "schema_valid": not structural_errors,
        "ready_for_six_family_pilot": not blockers and not environment_blockers,
        "strict_mode": args.strict,
        "note": (
            "Repo/CI validity is intentionally weaker than pilot readiness. "
            "Pending locks are expected before local dataset freeze and must not be bypassed."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if structural_errors:
        return 2
    if args.strict and (blockers or environment_blockers):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
