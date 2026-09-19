from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

import torch


EXPECTED = {
    "transformers": "4.44.2",
    "accelerate": "0.34.2",
}


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
        "errors": errors,
    }

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "vram_gb": round(props.total_memory / (1024**3), 3),
            }
        )

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
