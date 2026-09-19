from __future__ import annotations

import argparse
import json
from pathlib import Path

from faireval.freeze import canonical_json, file_sha256
from faireval.local_plan import compile_local_open_weight_plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile the two-model local open-weight FairEval plan"
    )
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument("--counterfactuals", default="configs/counterfactuals.yaml")
    parser.add_argument("--models", default="configs/local_models.yaml")
    parser.add_argument("--output-dir", default="results/plans/local-open-weight-v1")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--fairsynth-users", type=int, default=360)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--no-real-world", action="store_true")
    args = parser.parse_args()

    cells, manifest = compile_local_open_weight_plan(
        freeze_root=Path(args.freeze_root),
        counterfactuals_yaml=Path(args.counterfactuals),
        local_models_yaml=Path(args.models),
        seed=args.seed,
        include_real_world=not args.no_real_world,
        fairsynth_users=args.fairsynth_users,
        repetitions=args.repetitions,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "run_plan.jsonl"
    plan_path.write_text(
        "".join(canonical_json(row) + "\n" for row in cells),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        **manifest,
        "plan_sha256": __import__("hashlib").sha256(
            canonical_json(cells).encode("utf-8")
        ).hexdigest(),
        "run_plan_file_sha256": file_sha256(plan_path),
    }
    (output / "plan_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
