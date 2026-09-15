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

SECRET_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "LLAMA_PROVIDER_API_KEY",
)

ENDPOINT_ENV_VARS = (
    "QWEN_BASE_URL",
    "LLAMA_BASE_URL",
    "LLAMA_PROVIDER_NAME",
)

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


def _load_models(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = payload.get("models", []) if isinstance(payload, dict) else []
    if not isinstance(models, list):
        raise ValueError("configs/models.yaml must contain a models list")
    return [dict(row) for row in models if isinstance(row, dict) and row.get("enabled", True)]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "configs" / "models.yaml"
    models = _load_models(model_path)

    package_versions = {
        label: _version(distribution) for label, distribution in PACKAGE_NAMES.items()
    }
    missing_packages = sorted(label for label, version in package_versions.items() if version is None)

    credentials = {name: bool(os.environ.get(name)) for name in SECRET_ENV_VARS}
    endpoints = {name: os.environ.get(name) or None for name in ENDPOINT_ENV_VARS}

    model_rows = []
    model_errors = []
    for row in models:
        family = str(row.get("family", "")).strip()
        model_id = str(row.get("model_id", "")).strip()
        reasoning = str(row.get("reasoning_or_thinking_setting", "")).strip()
        sampling = str(row.get("sampling_policy", "")).strip()
        token_parameter = str(row.get("output_token_parameter", "")).strip()
        if not all((family, model_id, reasoning, sampling, token_parameter)):
            model_errors.append(f"incomplete model policy for family={family or '<missing>'}")
        if token_parameter and token_parameter not in VALID_OUTPUT_TOKEN_PARAMETERS:
            model_errors.append(
                f"unrecognized output_token_parameter for {family}: {token_parameter}"
            )
        model_rows.append(
            {
                "family": family,
                "model_id": model_id,
                "reasoning_or_thinking_setting": reasoning,
                "sampling_policy": sampling,
                "output_token_parameter": token_parameter,
            }
        )

    if len(model_rows) != 6:
        model_errors.append(f"expected 6 enabled model families, found {len(model_rows)}")

    blockers = []
    if sys.version_info < (3, 10):
        blockers.append("Python 3.10+ is required")
    blockers.extend(f"missing package: {name}" for name in missing_packages)
    blockers.extend(model_errors)
    if not endpoints["QWEN_BASE_URL"]:
        blockers.append("QWEN_BASE_URL is not frozen")
    if not endpoints["LLAMA_BASE_URL"]:
        blockers.append("LLAMA_BASE_URL is not frozen")
    if not endpoints["LLAMA_PROVIDER_NAME"]:
        blockers.append("LLAMA_PROVIDER_NAME is not frozen")

    report = {
        "schema_version": "faireval-environment-preflight-v2",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions,
        "credential_presence_only": credentials,
        "endpoints": endpoints,
        "model_panel": model_rows,
        "blockers": blockers,
        "ready_for_offline_tests": not missing_packages and not model_errors,
        "ready_for_six_family_api_pilot": not blockers and all(credentials.values()),
        "note": "No credential values are printed; only presence/absence is reported.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready_for_offline_tests"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
