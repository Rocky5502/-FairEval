from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import yaml

from faireval.providers.factory import PHI35_LOCAL_REVISION, QWEN25_LOCAL_REVISION


ROOT = Path(__file__).resolve().parents[1]


def _load_revisions() -> dict[str, str]:
    payload = yaml.safe_load((ROOT / "configs" / "local_models.yaml").read_text(encoding="utf-8"))
    rows = payload.get("models", []) if isinstance(payload, dict) else []
    return {
        str(row.get("family")): str(row.get("revision", ""))
        for row in rows
        if isinstance(row, dict) and row.get("enabled", True)
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check FairEval local open-weight GPU/revision readiness without loading model weights"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero unless CUDA is available and frozen revisions match the provider factory",
    )
    args = parser.parse_args()

    revisions = _load_revisions()
    revision_checks = {
        "qwen25_local": revisions.get("qwen25_local") == QWEN25_LOCAL_REVISION,
        "phi35_local": revisions.get("phi35_local") == PHI35_LOCAL_REVISION,
    }
    report: dict[str, Any] = {
        "schema_version": "faireval-local-gpu-preflight-v2",
        "platform": platform.platform(),
        "canonical_target": "NVIDIA GeForce RTX 5090 32GB or larger compatible CUDA GPU",
        "qwen25_revision": revisions.get("qwen25_local"),
        "phi35_revision": revisions.get("phi35_local"),
        "revision_contract_ok": all(revision_checks.values()),
        "local_dtype": "bfloat16",
        "cuda_available": False,
        "gpu": None,
        "warnings": [],
    }

    try:
        import torch
    except ImportError:
        report["warnings"].append(
            "PyTorch is not installed; install the CUDA-matched PyTorch build before local execution."
        )
        torch = None

    if torch is not None:
        report["torch_version"] = torch.__version__
        report["cuda_runtime_version"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "vram_bytes": int(props.total_memory),
                "vram_gib": round(int(props.total_memory) / (1024**3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
            }
            if int(props.total_memory) < 24 * 1024**3:
                report["warnings"].append(
                    "GPU has under 24 GiB VRAM; canonical BF16 local runs may require a different GPU or separately versioned quantized protocol."
                )
        else:
            report["warnings"].append("CUDA is not available to PyTorch.")

    if not report["revision_contract_ok"]:
        report["warnings"].append(
            "Frozen local-model revisions disagree between configs/local_models.yaml and provider factory."
        )

    report["ready_for_local_pilot"] = bool(
        report["cuda_available"] and report["revision_contract_ok"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.strict and not report["ready_for_local_pilot"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
