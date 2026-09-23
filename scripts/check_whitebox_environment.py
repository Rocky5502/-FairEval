from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch


EXPECTED = {
    "transformers": "4.44.2",
    "accelerate": "0.34.2",
    "bitsandbytes": "0.50.2",
    "sentencepiece": "0.2.2",
    "protobuf": "7.36.2",
}
EXPECTED_PYTHON = "3.11.9"
EXPECTED_TORCH = "2.11.0+cu128"


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    packages = {
        "transformers": _version("transformers"),
        "accelerate": _version("accelerate"),
        "bitsandbytes": _version("bitsandbytes"),
        "sentencepiece": _version("sentencepiece"),
        "protobuf": _version("protobuf"),
    }

    errors: list[str] = []
    if platform.python_version() != EXPECTED_PYTHON:
        errors.append(
            f"Python must be {EXPECTED_PYTHON}; found {platform.python_version()!r}"
        )
    if str(torch.__version__) != EXPECTED_TORCH:
        errors.append(f"torch must be {EXPECTED_TORCH}; found {str(torch.__version__)!r}")
    for name, expected in EXPECTED.items():
        if packages[name] != expected:
            errors.append(f"{name} must be {expected}; found {packages[name]!r}")

    if packages["bitsandbytes"] is None:
        errors.append("bitsandbytes is not installed")

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        errors.append("CUDA is not available")

    result = {
        "schema_version": "faireval-whitebox-environment-v2",
        "status": "PASS" if not errors else "FAIL",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "packages": packages,
        "cuda_available": cuda_available,
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET"),
        "errors": errors,
    }

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        compute_capability = torch.cuda.get_device_capability(0)
        result.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "compute_capability": list(compute_capability),
                "vram_gb": round(props.total_memory / (1024**3), 3),
                "bnb_cuda_version_override": os.environ.get("BNB_CUDA_VERSION"),
            }
        )
        if torch.version.cuda != "12.8":
            errors.append(
                f"canonical local runtime requires PyTorch CUDA 12.8; found {torch.version.cuda!r}"
            )
        if tuple(compute_capability) < (12, 0):
            errors.append(
                f"expected Blackwell-class compute capability >= 12.0; found {compute_capability}"
            )
        if os.environ.get("BNB_CUDA_VERSION") != "128":
            errors.append(
                "BNB_CUDA_VERSION must be exactly 128 for the canonical V5 CUDA 12.8 runtime"
            )
        if os.environ.get("HF_HUB_DISABLE_XET") != "1":
            errors.append("HF_HUB_DISABLE_XET must be exactly 1 for the canonical local run")
        result["status"] = "PASS" if not errors else "FAIL"
        result["errors"] = errors

    output = Path("results/whitebox/environment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
