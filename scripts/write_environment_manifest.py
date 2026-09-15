from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = (
    "configs/datasets.yaml",
    "configs/counterfactuals.yaml",
    "configs/prompt_suite.yaml",
    "configs/cue_suite.yaml",
    "configs/models.yaml",
    "configs/experiment.yaml",
    "configs/study_design.yaml",
)
PACKAGES = (
    "numpy",
    "pandas",
    "openpyxl",
    "PyYAML",
    "scipy",
    "statsmodels",
    "matplotlib",
    "openai",
    "anthropic",
    "google-genai",
)
SECRET_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "LLAMA_PROVIDER_API_KEY",
)
ENDPOINT_ENV = (
    "QWEN_BASE_URL",
    "LLAMA_BASE_URL",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest() -> dict:
    config_hashes = {}
    for relative in CONFIG_FILES:
        path = ROOT / relative
        config_hashes[relative] = _file_hash(path) if path.is_file() else None

    endpoint_fingerprints = {}
    for name in ENDPOINT_ENV:
        value = os.environ.get(name)
        endpoint_fingerprints[name] = {
            "set": bool(value),
            "sha256": None if not value else _sha256_bytes(value.encode("utf-8")),
        }

    return {
        "schema_version": "faireval-environment-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_sha(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "config_sha256": config_hashes,
        "credential_presence": {name: bool(os.environ.get(name)) for name in SECRET_ENV},
        "endpoint_fingerprints": endpoint_fingerprints,
        "llama_provider_name": os.environ.get("LLAMA_PROVIDER_NAME"),
        "privacy_note": (
            "Credential values and raw provider endpoint URLs are never written; "
            "only presence flags and endpoint SHA-256 fingerprints are recorded."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a non-secret FairEval environment manifest")
    parser.add_argument("--output", default="results/environment_manifest.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_manifest()
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"git_commit_sha={payload['git_commit_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
