from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_BASE_URL = "https://api.zhizengzeng.com/v1"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _json_request(url: str, *, api_key: str, method: str = "GET") -> tuple[int, bytes]:
    data = b"{}" if method == "POST" else None
    request = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "FairEval-ECIR2027-gateway-preflight/2.0",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return int(getattr(response, "status", 200)), response.read()


def _model_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("GET /models did not return an OpenAI-style data list")
    return sorted(
        {
            str(row["id"])
            for row in payload["data"]
            if isinstance(row, dict) and row.get("id") is not None
        },
        key=str.lower,
    )


def _frozen_models(path: Path) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("models config lacks a models list")
    return {
        str(row["family"]): str(row["model_id"])
        for row in rows
        if isinstance(row, dict) and row.get("enabled", True)
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-generation Zhizengzeng preflight: verify exact frozen model IDs "
            "and account balance without printing the API key."
        )
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--output-dir", default="results/preflight")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    _load_dotenv(Path(args.env_file))
    base_url = os.environ.get("ZZZ_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    api_key = os.environ.get("ZZZ_API_KEY")
    if not api_key:
        raise RuntimeError("ZZZ_API_KEY is required; its value is never printed")

    frozen = _frozen_models(Path(args.models))
    try:
        model_status, model_raw = _json_request(
            f"{base_url}/models", api_key=api_key, method="GET"
        )
        balance_status, balance_raw = _json_request(
            f"{base_url}/dashboard/billing/credit_grants",
            api_key=api_key,
            method="POST",
        )
    except urllib.error.HTTPError as exc:
        preview = exc.read(1000).decode("utf-8", errors="replace")
        print(
            json.dumps(
                {
                    "status": "HTTP_ERROR",
                    "http_status": exc.code,
                    "response_preview": preview,
                    "api_key_value_printed": False,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 3

    model_payload = json.loads(model_raw.decode("utf-8"))
    ids = _model_ids(model_payload)
    available = set(ids)
    exact = {family: model_id in available for family, model_id in frozen.items()}
    missing = {family: model_id for family, model_id in frozen.items() if model_id not in available}

    balance_payload = json.loads(balance_raw.decode("utf-8"))
    try:
        balance_rmb = float(balance_payload["grants"]["available_amount"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Unexpected balance response schema") from exc

    now = datetime.now(timezone.utc)
    report = {
        "schema_version": "faireval-zhizengzeng-preflight-v2",
        "status": "PASS" if not missing else "BLOCKED_MODEL_ID_MISMATCH",
        "scope": "model_list_and_balance_only_no_generation",
        "gateway": "zhizengzeng",
        "base_url": base_url,
        "utc": now.isoformat(),
        "models_http_status": model_status,
        "balance_http_status": balance_status,
        "frozen_models": frozen,
        "exact_model_id_checks": exact,
        "missing_frozen_models": missing,
        "available_balance_rmb": balance_rmb,
        "model_count": len(ids),
        "model_ids_sha256": hashlib.sha256(
            ("\n".join(ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "api_key_value_printed": False,
        "paid_generation_calls_made": 0,
        "guard": (
            "Do not execute the hosted plan if any frozen model ID is absent. "
            "Version the model manifest instead of silently substituting a model."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"zhizengzeng_preflight_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["artifact"] = str(path)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))

    if args.strict and missing:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
