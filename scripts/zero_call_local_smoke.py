from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from faireval.datasets.fairsynth360 import FairSynth360Adapter
from faireval.execute import load_and_verify_plan, pending_cells
from faireval.freeze import canonical_json, file_sha256, freeze_dataset, verify_freeze
from faireval.local_plan import compile_local_open_weight_plan


EXPECTED_FAMILIES = {"qwen25_local", "phi35_local"}


def _write_plan(cells: list[dict[str, object]], manifest: dict[str, object], output_dir: Path) -> None:
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


def run_smoke(repo_root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="faireval-local-zero-call-") as tmp:
        root = Path(tmp)
        freeze_root = root / "frozen"
        synth_dir = freeze_root / "fairsynth360"
        plan_dir = root / "plan"

        freeze_manifest = freeze_dataset(
            FairSynth360Adapter(),
            root / "unused-raw",
            synth_dir,
            users=12,
            candidate_set_size=30,
            max_history_items=8,
            seed=1729,
        )
        verified_freeze = verify_freeze(synth_dir)

        cells, manifest = compile_local_open_weight_plan(
            freeze_root=freeze_root,
            counterfactuals_yaml=repo_root / "configs" / "counterfactuals.yaml",
            local_models_yaml=repo_root / "configs" / "local_models.yaml",
            seed=1729,
            include_real_world=False,
            fairsynth_users=12,
            repetitions=1,
        )
        _write_plan(cells, manifest, plan_dir)
        verified_cells, verified_manifest = load_and_verify_plan(plan_dir)
        pending = pending_cells(verified_cells, completed=set())

        families = {str(row["model_family"]) for row in verified_cells}
        if families != EXPECTED_FAMILIES:
            raise AssertionError(f"unexpected local families: {sorted(families)}")
        if len(verified_cells) != 144:
            raise AssertionError(
                "12 FairSynth users x 6 registered conditions x 2 local models should yield 144 cells; "
                f"got {len(verified_cells)}"
            )
        if len(pending) != len(verified_cells):
            raise AssertionError("zero-call dry run must leave every planned cell pending")
        if any(row.get("seed") is None for row in verified_cells):
            raise AssertionError("every local open-weight cell must have a frozen generation seed")
        if any(row.get("output_token_parameter") != "max_new_tokens" for row in verified_cells):
            raise AssertionError("every local cell must freeze max_new_tokens semantics")
        if any(str(row.get("dataset")) != "fairsynth360" for row in verified_cells):
            raise AssertionError("zero-call smoke must contain only FairSynth cells")

        return {
            "schema_version": "faireval-local-zero-call-smoke-v1",
            "status": "PASS",
            "api_calls_made": 0,
            "model_weights_loaded": False,
            "providers_instantiated": False,
            "dataset": "fairsynth360",
            "frozen_users": int(verified_freeze["verified_instance_count"]),
            "freeze_instances_sha256": str(verified_freeze["verified_instances_sha256"]),
            "model_families": sorted(families),
            "planned_cells": len(verified_cells),
            "pending_cells": len(pending),
            "all_local_generation_seeds_frozen": True,
            "output_token_parameter": "max_new_tokens",
            "plan_sha256": str(verified_manifest["plan_sha256"]),
            "run_plan_file_sha256": str(verified_manifest["run_plan_file_sha256"]),
            "local_models_yaml_sha256": str(verified_manifest["local_models_yaml_sha256"]),
            "counterfactuals_yaml_sha256": str(verified_manifest["counterfactuals_yaml_sha256"]),
            "freeze_manifest_schema": str(freeze_manifest["schema_version"]),
            "note": (
                "This smoke stops before provider construction, model loading, CUDA use, or any API call. "
                "It validates the project-owned FairSynth freeze and immutable two-model local plan path."
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run FairEval FairSynth -> local-plan -> dry-run verification with zero provider calls."
    )
    parser.add_argument(
        "--output",
        default="results/smoke/local_zero_call_v1.json",
        help="JSON summary path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    report = run_smoke(repo_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
