from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from faireval.freeze import canonical_json, file_sha256
from faireval.hosted_plan import compile_hosted_fairsynth_plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a deterministic, identity-balanced FairSynth-360 plan for "
            "the six hosted gateway families. No API call is made."
        )
    )
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--output-dir", default="results/plans/hosted-fairsynth-budget-v1")
    parser.add_argument("--users", type=int, default=120)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    args = parser.parse_args()

    cells, manifest = compile_hosted_fairsynth_plan(
        freeze_root=Path(args.freeze_root),
        models_yaml=Path(args.models),
        users=args.users,
        repetitions=args.repetitions,
        seed=args.seed,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "run_plan.jsonl"
    plan_path.write_text(
        "".join(canonical_json(row) + "\n" for row in cells),
        encoding="utf-8",
        newline="\n",
    )
    payload = {
        **manifest,
        "plan_sha256": hashlib.sha256(canonical_json(cells).encode("utf-8")).hexdigest(),
        "run_plan_file_sha256": file_sha256(plan_path),
    }
    (output_dir / "plan_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
