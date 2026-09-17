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
    rows = [
        row
        for row in models.get("models", [])
        if isinstance(row, dict) and row.get("enabled", True)
    ]
    if len(rows) != 6:
        blockers.append(f"model panel must contain six enabled families; found {len(rows)}")
        return blockers
    by_family = {str(row.get("family")): row for row in rows}
    expected = {"openai", "anthropic", "google", "deepseek", "qwen", "meta"}
    if set(by_family) != expected:
        blockers.append(f"unexpected enabled model families: {sorted(by_family)}")

    gateway = models.get("hosted_gateway")
    if not isinstance(gateway, dict) or gateway.get("name") != "zhizengzeng":
        blockers.append("hosted_gateway: expected frozen zhizengzeng configuration")

    for family, row in by_family.items():
        if row.get("provider") != "zhizengzeng":
            blockers.append(f"{family}: provider is not frozen to zhizengzeng")
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
    if os.environ.get("FAIREVAL_HOSTED_GATEWAY", "").strip().lower() != "zhizengzeng":
        blockers.append("environment: FAIREVAL_HOSTED_GATEWAY must be zhizengzeng")
    if not os.environ.get("ZZZ_API_KEY"):
        blockers.append("environment: ZZZ_API_KEY missing")
    base_url = os.environ.get("ZZZ_BASE_URL", "https://api.zhizengzeng.com/v1")
    if not base_url.startswith("https://api.zhizengzeng.com/"):
        blockers.append("environment: ZZZ_BASE_URL is not the frozen Zhizengzeng host")
    return blockers


def _budget_blockers(repo_root: Path) -> list[str]:
    path = repo_root / "configs" / "hosted_budget.yaml"
    if not path.is_file():
        return ["configs/hosted_budget.yaml missing"]
    budget = _load(path)
    policy = budget.get("policy")
    if not isinstance(policy, dict):
        return ["hosted budget policy mapping missing"]
    blockers: list[str] = []
    if float(policy.get("planning_target_rmb", -1)) != 200.0:
        blockers.append("hosted budget planning target is not frozen to 200 RMB")
    if float(policy.get("hard_cap_rmb", -1)) != 250.0:
        blockers.append("hosted budget emergency stop threshold is not frozen to 250 RMB")
    if float(policy.get("per_request_reserve_rmb", -1)) != 2.0:
        blockers.append("hosted budget request reserve is not frozen to 2 RMB")
    if policy.get("bypass_allowed") is not False:
        blockers.append("hosted budget policy must prohibit deliberate threshold bypass")
    if policy.get("provider_side_atomic_spend_cap_claimed") is not False:
        blockers.append("hosted budget policy must not claim an unverified provider-side atomic cap")
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate whether FairEval is safe to promote to a budgeted hosted API pilot."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero until dataset locks, gateway freeze, budget policy and credentials are complete.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_lock_path = repo_root / "configs" / "dataset_releases.yaml"
    models_path = repo_root / "configs" / "models.yaml"

    dataset_lock = _load(dataset_lock_path)
    models = _load(models_path)
    blockers = (
        _dataset_blockers(repo_root, dataset_lock)
        + _model_blockers(models)
        + _budget_blockers(repo_root)
    )
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
        "schema_version": "faireval-pilot-readiness-v3",
        "dataset_ids": list(DATASET_IDS),
        "hosted_gateway": "zhizengzeng",
        "budget_target_rmb": 200.0,
        "budget_emergency_stop_threshold_rmb": 250.0,
        "provider_side_atomic_spend_cap_claimed": False,
        "dataset_model_budget_blockers": blockers,
        "environment_blockers": environment_blockers,
        "schema_valid": not structural_errors,
        "ready_for_six_family_pilot": not blockers and not environment_blockers,
        "strict_mode": args.strict,
        "note": (
            "Repo/CI validity is intentionally weaker than paid pilot readiness. "
            "Third-party dataset release locks remain mandatory. Hosted execution "
            "uses the unified gateway plus a client-side, balance-reconciled 200 RMB "
            "normal stop and 250 RMB emergency stop threshold; no provider-side atomic "
            "spend cap is claimed without independent verification."
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
