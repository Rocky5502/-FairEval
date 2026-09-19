from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ZZZ_BASE_URL = "https://api.zhizengzeng.com/v1"


def frozen_hosted_model_ids(models_yaml: Path) -> dict[str, str]:
    payload = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    rows = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{models_yaml} lacks a models list")
    result = {
        str(row["family"]): str(row["model_id"])
        for row in rows
        if isinstance(row, dict) and row.get("enabled", True)
    }
    expected = {"openai", "anthropic", "google", "deepseek", "qwen", "meta"}
    if set(result) != expected:
        raise ValueError(
            f"hosted model manifest must contain exactly six families: {sorted(expected)}"
        )
    return result


def fetch_gateway_model_ids(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: int = 45,
) -> list[str]:
    base = (base_url or os.environ.get("ZZZ_BASE_URL") or DEFAULT_ZZZ_BASE_URL).rstrip("/")
    request = urllib.request.Request(
        f"{base}/models",
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "FairEval-ECIR2027-model-gate/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    payload: Any = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError("Zhizengzeng GET /models returned an unexpected schema")
    return sorted(
        {
            str(row["id"])
            for row in payload["data"]
            if isinstance(row, dict) and row.get("id") is not None
        },
        key=str.lower,
    )


def verify_frozen_gateway_models(
    *,
    models_yaml: Path,
    api_key: str,
    base_url: str | None = None,
) -> dict[str, Any]:
    frozen = frozen_hosted_model_ids(models_yaml)
    available = set(fetch_gateway_model_ids(api_key=api_key, base_url=base_url))
    exact = {family: model_id in available for family, model_id in frozen.items()}
    missing = {family: model_id for family, model_id in frozen.items() if model_id not in available}
    return {
        "frozen_models": frozen,
        "exact_model_id_checks": exact,
        "missing_frozen_models": missing,
        "all_exact_models_available": not missing,
    }
