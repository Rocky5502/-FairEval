from __future__ import annotations

import argparse
import json
from pathlib import Path

from faireval.robustness_plan import compile_rq3_extra_plan, write_rq3_plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile the factorized FairEval RQ3 extra-cell plan; makes zero API calls"
    )
    parser.add_argument("--freeze-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--counterfactuals", default="configs/counterfactuals.yaml")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--study-design", default="configs/study_design.yaml")
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    cells, manifest = compile_rq3_extra_plan(
        freeze_root=Path(args.freeze_root),
        counterfactuals_yaml=Path(args.counterfactuals),
        models_yaml=Path(args.models),
        study_design_yaml=Path(args.study_design),
        experiment_yaml=Path(args.experiment),
        seed=args.seed,
    )
    paths = write_rq3_plan(Path(args.output_dir), cells, manifest)
    result = {
        **manifest,
        "api_calls_made": 0,
        "run_plan": str(paths["plan"]),
        "plan_manifest": str(paths["manifest"]),
        "instruction": (
            "Inspect this factorized extra-cell plan and the baseline-reuse manifest before "
            "executing it with faireval execute-plan."
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
