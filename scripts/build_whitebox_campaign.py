from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from faireval.datasets.fairsynth360 import FairSynth360Adapter
from faireval.freeze import canonical_json, file_sha256, freeze_dataset, verify_freeze
from faireval.local_plan import compile_local_open_weight_plan


EXPECTED_FAIRSYNTH_USERS = 360
EXPECTED_CONDITIONS_PER_USER = 6
EXPECTED_REPETITIONS = 3
EXPECTED_MODEL_COUNT = 2
EXPECTED_FAIRSYNTH_CORE_CELLS = (
    EXPECTED_FAIRSYNTH_USERS
    * EXPECTED_CONDITIONS_PER_USER
    * EXPECTED_REPETITIONS
    * EXPECTED_MODEL_COUNT
)


def _write_plan(output_dir: Path, cells: list[dict], manifest: dict) -> dict:
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
    return payload


def _ensure_fairsynth(freeze_root: Path, *, rebuild: bool) -> dict:
    target = freeze_root / "fairsynth360"
    if rebuild or not (target / "manifest.json").is_file():
        freeze_dataset(
            FairSynth360Adapter(),
            Path("."),
            target,
            users=EXPECTED_FAIRSYNTH_USERS,
            candidate_set_size=30,
            max_history_items=8,
            seed=1729,
        )
    result = verify_freeze(target)
    if result.get("verification") != "PASS":
        raise RuntimeError("FairSynth-360 freeze verification failed")
    if int(result.get("verified_instance_count", -1)) != EXPECTED_FAIRSYNTH_USERS:
        raise RuntimeError(
            "canonical white-box campaign requires exactly 360 FairSynth users; "
            f"found {result.get('verified_instance_count')}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the full FairEval two-model white-box campaign. This planner "
            "loads no model weights and makes no hosted API calls."
        )
    )
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument("--output-root", default="results/plans/whitebox-full-v1")
    parser.add_argument("--counterfactuals", default="configs/counterfactuals.yaml")
    parser.add_argument("--models", default="configs/local_models.yaml")
    parser.add_argument("--campaign", default="configs/whitebox_campaign.yaml")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--include-real-world", action="store_true")
    parser.add_argument("--rebuild-fairsynth", action="store_true")
    args = parser.parse_args()

    campaign_cfg = yaml.safe_load(Path(args.campaign).read_text(encoding="utf-8"))
    if not isinstance(campaign_cfg, dict):
        raise ValueError("whitebox campaign config must be a YAML mapping")
    if int(campaign_cfg.get("seed", -1)) != args.seed:
        raise ValueError("campaign seed and CLI seed disagree")

    freeze_root = Path(args.freeze_root)
    verified = _ensure_fairsynth(freeze_root, rebuild=args.rebuild_fairsynth)

    cells, manifest = compile_local_open_weight_plan(
        freeze_root=freeze_root,
        counterfactuals_yaml=Path(args.counterfactuals),
        local_models_yaml=Path(args.models),
        seed=args.seed,
        include_real_world=args.include_real_world,
        fairsynth_users=EXPECTED_FAIRSYNTH_USERS,
        repetitions=EXPECTED_REPETITIONS,
    )

    synth_cells = [row for row in cells if str(row.get("dataset")) == "fairsynth360"]
    if len(synth_cells) != EXPECTED_FAIRSYNTH_CORE_CELLS:
        raise AssertionError(
            "FairSynth full-plan geometry drift: expected "
            f"{EXPECTED_FAIRSYNTH_CORE_CELLS}, got {len(synth_cells)}"
        )
    families = {str(row["model_family"]) for row in synth_cells}
    if families != {"qwen25_local", "phi35_local"}:
        raise AssertionError(f"unexpected local families: {sorted(families)}")
    if any(row.get("seed") is None for row in cells):
        raise AssertionError("every white-box cell must have a frozen generation seed")
    if any(row.get("output_token_parameter") != "max_new_tokens" for row in cells):
        raise AssertionError("white-box cells must use max_new_tokens")

    output_root = Path(args.output_root)
    payload = _write_plan(
        output_root / "core",
        cells,
        {
            **manifest,
            "schema_version": "faireval-whitebox-full-plan-v1",
            "campaign_config_sha256": file_sha256(Path(args.campaign)),
            "fairsynth_verified_instances_sha256": verified["verified_instances_sha256"],
            "fairsynth_expected_core_cells": EXPECTED_FAIRSYNTH_CORE_CELLS,
            "real_world_included": bool(args.include_real_world),
            "execution_policy": {
                "one_model_family_per_process": True,
                "recommended_resume_batch_cells": 250,
                "model_weights_loaded_by_planner": False,
                "hosted_api_calls_made_by_planner": 0,
            },
        },
    )

    summary = {
        "status": "READY",
        "campaign": campaign_cfg.get("version"),
        "fairsynth_users": EXPECTED_FAIRSYNTH_USERS,
        "fairsynth_core_cells": len(synth_cells),
        "all_core_cells": len(cells),
        "model_families": sorted(families),
        "real_world_included": bool(args.include_real_world),
        "plan_dir": str(output_root / "core"),
        "plan_sha256": payload["plan_sha256"],
        "next": (
            "Run scripts/run_whitebox_family.py separately for qwen25_local and "
            "phi35_local. Start with a small --max-cells canary, audit, then resume."
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
