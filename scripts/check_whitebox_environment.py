import json
import platform
from pathlib import Path

import torch


def main():
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)

        result.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "vram_gb": round(
                    props.total_memory / (1024 ** 3),
                    3,
                ),
            }
        )

    output = Path(
        "results/whitebox/environment.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()