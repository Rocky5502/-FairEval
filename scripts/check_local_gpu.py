from __future__ import annotations

import argparse
import json
import os
import platform
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check FairEval local open-weight GPU/revision readiness without loading model weights"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero unless CUDA and both exact local model revisions are frozen",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": "faireval-local-gpu-preflight-v1",
        "platform": platform.platform(),
        "canonical_target": "NVIDIA GeForce RTX 5090 32GB or larger compatible CUDA GPU",
        "qwen25_revision_frozen": bool(os.environ.get("QWEN25_LOCAL_REVISION")),
        "phi35_revision_frozen": bool(os.environ.get("PHI35_LOCAL_REVISION")),
        "local_dtype": os.environ.get("FAIREVAL_LOCAL_DTYPE", "bfloat16"),
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

    if not report["qwen25_revision_frozen"]:
        report["warnings"].append("QWEN25_LOCAL_REVISION is not frozen yet.")
    if not report["phi35_revision_frozen"]:
        report["warnings"].append("PHI35_LOCAL_REVISION is not frozen yet.")

    report["ready_for_local_pilot"] = bool(
        report["cuda_available"]
        and report["qwen25_revision_frozen"]
        and report["phi35_revision_frozen"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.strict and not report["ready_for_local_pilot"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
