from __future__ import annotations

import argparse
import json
from pathlib import Path

from faireval.local_robustness_plan import compile_local_rq3_extra_plan
from faireval.robustness_plan import write_rq3_plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the registered FairEval RQ3 robustness cells for the two "
            "frozen local white-box models. Makes zero model/API calls."
        )
    )
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument(
        "--output-dir",
        default="results/plans/whitebox-full-v1/rq3",
    )
    parser.add_argument("--counterfactuals", default="configs/counterfactuals.yaml")
    parser.add_argument("--models", default="configs/local_models.yaml")
    parser.add_argument("--study-design", default="configs/study_design.yaml")
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    cells, manifest = compile_local_rq3_extra_plan(
        freeze_root=Path(args.freeze_root),
        counterfactuals_yaml=Path(args.counterfactuals),
        local_models_yaml=Path(args.models),
        study_design_yaml=Path(args.study_design),
        experiment_yaml=Path(args.experiment),
        seed=args.seed,
    )
    paths = write_rq3_plan(Path(args.output_dir), cells, manifest)
    result = {
        **manifest,
        "model_or_api_calls_made": 0,
        "run_plan": str(paths["plan"]),
        "plan_manifest": str(paths["manifest"]),
        "next": (
            "Dry-run each family with scripts/run_whitebox_family.py using this "
            "plan directory, then execute in resume-safe batches after audit."
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
