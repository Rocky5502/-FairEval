from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import yaml


PACKAGE_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "openpyxl": "openpyxl",
    "yaml": "PyYAML",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "openai": "openai",
    "anthropic": "anthropic",
    "google_genai": "google-genai",
    "pytest": "pytest",
    "ruff": "ruff",
}

VALID_OUTPUT_TOKEN_PARAMETERS = {
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
}


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_models(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configs/models.yaml must contain a mapping")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("configs/models.yaml must contain a models list")
    return payload, [
        dict(row)
        for row in models
        if isinstance(row, dict) and row.get("enabled", True)
    ]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "configs" / "models.yaml"
    models_payload, models = _load_models(model_path)

    package_versions = {
        label: _version(distribution) for label, distribution in PACKAGE_NAMES.items()
    }
    # Native anthropic/google packages remain optional when the unified gateway is used.
    core_required_packages = {
        "numpy",
        "pandas",
        "openpyxl",
        "yaml",
        "scipy",
        "statsmodels",
        "matplotlib",
        "openai",
        "pytest",
        "ruff",
    }
    missing_packages = sorted(
        label
        for label in core_required_packages
        if package_versions.get(label) is None
    )

    gateway = models_payload.get("hosted_gateway")
    gateway_errors: list[str] = []
    if not isinstance(gateway, dict) or gateway.get("name") != "zhizengzeng":
        gateway_errors.append("configs/models.yaml hosted_gateway is not frozen to zhizengzeng")

    gateway_mode = os.environ.get("FAIREVAL_HOSTED_GATEWAY", "").strip().lower()
    gateway_key_present = bool(os.environ.get("ZZZ_API_KEY"))
    gateway_base_url = os.environ.get("ZZZ_BASE_URL", "https://api.zhizengzeng.com/v1")
    gateway_base_url_ok = gateway_base_url.rstrip("/") == "https://api.zhizengzeng.com/v1"

    model_rows = []
    model_errors = []
    for row in models:
        family = str(row.get("family", "")).strip()
        model_id = str(row.get("model_id", "")).strip()
        reasoning = str(row.get("reasoning_or_thinking_setting", "")).strip()
        sampling = str(row.get("sampling_policy", "")).strip()
        token_parameter = str(row.get("output_token_parameter", "")).strip()
        provider = str(row.get("provider", "")).strip()
        if not all((family, model_id, reasoning, sampling, token_parameter, provider)):
            model_errors.append(f"incomplete model policy for family={family or '<missing>'}")
        if provider and provider != "zhizengzeng":
            model_errors.append(f"{family}: provider is not frozen to zhizengzeng")
        if token_parameter and token_parameter not in VALID_OUTPUT_TOKEN_PARAMETERS:
            model_errors.append(
                f"unrecognized output_token_parameter for {family}: {token_parameter}"
            )
        model_rows.append(
            {
                "family": family,
                "model_id": model_id,
                "provider": provider,
                "reasoning_or_thinking_setting": reasoning,
                "sampling_policy": sampling,
                "output_token_parameter": token_parameter,
                "gateway_wire_output_token_parameter": row.get(
                    "gateway_wire_output_token_parameter"
                ),
            }
        )

    if len(model_rows) != 6:
        model_errors.append(f"expected 6 enabled model families, found {len(model_rows)}")

    blockers = []
    if sys.version_info < (3, 10):
        blockers.append("Python 3.10+ is required")
    blockers.extend(f"missing package: {name}" for name in missing_packages)
    blockers.extend(model_errors)
    blockers.extend(gateway_errors)
    if gateway_mode != "zhizengzeng":
        blockers.append("FAIREVAL_HOSTED_GATEWAY must be zhizengzeng for hosted execution")
    if not gateway_key_present:
        blockers.append("ZZZ_API_KEY is missing")
    if not gateway_base_url_ok:
        blockers.append("ZZZ_BASE_URL does not match the frozen Zhizengzeng base URL")

    report = {
        "schema_version": "faireval-environment-preflight-v3",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions,
        "gateway": {
            "name": "zhizengzeng",
            "mode_env": gateway_mode or None,
            "api_key_present": gateway_key_present,
            "base_url_set": bool(os.environ.get("ZZZ_BASE_URL")),
            "base_url_matches_frozen": gateway_base_url_ok,
        },
        "model_panel": model_rows,
        "blockers": blockers,
        "ready_for_offline_tests": not missing_packages and not model_errors and not gateway_errors,
        "ready_for_six_family_api_pilot": not blockers,
        "note": (
            "No credential values or raw endpoint values are printed. Native Anthropic/Google "
            "SDKs are optional in the current unified-gateway phase."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready_for_offline_tests"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
