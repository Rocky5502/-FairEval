import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_environment_manifest_never_serializes_raw_credentials_or_endpoints(tmp_path: Path):
    output = tmp_path / "environment.json"
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": "FAKE_SECRET_DO_NOT_WRITE",
            "ANTHROPIC_API_KEY": "FAKE_ANTHROPIC_SECRET",
            "QWEN_BASE_URL": "https://private.example.invalid/qwen-endpoint",
            "LLAMA_BASE_URL": "https://private.example.invalid/llama-endpoint",
            "LLAMA_PROVIDER_NAME": "test-provider",
        }
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "write_environment_manifest.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    raw = output.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert "FAKE_SECRET_DO_NOT_WRITE" not in raw
    assert "FAKE_ANTHROPIC_SECRET" not in raw
    assert "https://private.example.invalid/qwen-endpoint" not in raw
    assert "https://private.example.invalid/llama-endpoint" not in raw

    assert payload["credential_presence"]["OPENAI_API_KEY"] is True
    assert payload["credential_presence"]["ANTHROPIC_API_KEY"] is True
    assert payload["endpoint_fingerprints"]["QWEN_BASE_URL"]["set"] is True
    assert payload["endpoint_fingerprints"]["QWEN_BASE_URL"]["sha256"]
    assert payload["endpoint_fingerprints"]["LLAMA_BASE_URL"]["set"] is True
    assert payload["llama_provider_name"] == "test-provider"
