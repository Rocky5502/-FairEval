from __future__ import annotations

import argparse
import json
from pathlib import Path

from faireval.datasets.fairsynth360 import FairSynth360Adapter
from faireval.freeze import freeze_dataset, verify_freeze


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize the fully synthetic FairSynth-360 benchmark"
    )
    parser.add_argument("--output-dir", default="data/frozen/fairsynth360")
    parser.add_argument("--users", type=int, default=360)
    parser.add_argument("--candidate-set-size", type=int, default=30)
    parser.add_argument("--max-history-items", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    output = Path(args.output_dir)
    manifest = freeze_dataset(
        FairSynth360Adapter(),
        Path("."),  # ignored: FairSynth has no external raw input
        output,
        users=args.users,
        candidate_set_size=args.candidate_set_size,
        max_history_items=args.max_history_items,
        seed=args.seed,
    )
    verified = verify_freeze(output)
    if verified.get("verification") != "PASS":
        raise RuntimeError("FairSynth freeze verification did not pass")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
