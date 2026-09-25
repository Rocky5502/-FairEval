from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from faireval.analysis import aggregate_repetitions, score_run_log
from faireval.fairsynth_analysis import (
    build_fairsynth_identity_pairs,
    build_fairsynth_personality_pairs,
)
from faireval.freeze import canonical_json, file_sha256
from faireval.inference import summarize_paired_estimands


FAMILIES = ("openai", "anthropic", "google", "deepseek", "qwen", "meta")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the hosted paired-completion extension without mixing raw "
            "run logs from different execution commits."
        )
    )
    parser.add_argument(
        "--extension-plan-dir",
        default="results/plans/hosted-fairsynth-paired-extension-v1",
    )
    parser.add_argument(
        "--extension-run",
        default="results/runs/hosted-fairsynth-paired-extension-v1.jsonl",
    )
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument(
        "--output-dir",
        default="results/analysis/fairsynth-hosted-paired-v1",
    )
    parser.add_argument(
        "--overleaf-zip",
        default="dist/FairEval_ECIR2027_Overleaf_HOSTED_PAIRED.zip",
    )
    args = parser.parse_args()

    extension_manifest = _load_json(Path(args.extension_plan_dir) / "plan_manifest.json")
    target_records = extension_manifest.get("target_users")
    if not isinstance(target_records, list) or len(target_records) != 9:
        raise ValueError("hosted paired extension must define exactly 9 target users")
    target_users = {str(row["user_id"]) for row in target_records}
    if len(target_users) != 9:
        raise ValueError("target user IDs are not unique")

    if not all(bool(row.get("historically_untouched_user", False)) for row in target_records):
        raise ValueError("every hosted paired-extension target user must be historically untouched")

    extension_scored = score_run_log(
        Path(args.extension_run),
        plan_dir=Path(args.extension_plan_dir),
        freeze_root=Path(args.freeze_root),
    )
    combined = [
        row for row in extension_scored if str(row["user_id"]) in target_users
    ]
    extension_users_present = {str(row["user_id"]) for row in combined}
    if extension_users_present != target_users:
        raise RuntimeError(
            "extension hosted rows do not match the nine preregistered untouched users: "
            f"expected={sorted(target_users)}, actual={sorted(extension_users_present)}"
        )

    ids = [str(row["planned_cell_id"]) for row in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("target analysis contains duplicate planned cells")
    if len(combined) != 270:
        raise RuntimeError(
            "hosted paired target is incomplete: "
            f"expected 270 target cells, found {len(combined)}"
        )

    coverage: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in combined:
        coverage[(str(row["model_family"]), str(row["user_id"]))].add(
            str(row["condition_id"])
        )
    for family in FAMILIES:
        for user in target_users:
            conditions = coverage.get((family, user), set())
            has_required = (
                "C1" in conditions
                and "C3" in conditions
                and "C4" in conditions
                and sum(1 for value in conditions if value.startswith("C2:")) == 2
            )
            if not has_required or len(conditions) != 5:
                raise RuntimeError(
                    f"incomplete hosted target pairing for {family}/{user}: "
                    f"{sorted(conditions)}"
                )

    aggregated = aggregate_repetitions(combined)
    identity = build_fairsynth_identity_pairs(aggregated)
    personality = build_fairsynth_personality_pairs(aggregated)
    if len(identity) != 54 or len(personality) != 54:
        raise RuntimeError(
            f"expected 54 identity and 54 personality pairs, got "
            f"{len(identity)} and {len(personality)}"
        )

    inference = summarize_paired_estimands(
        identity + personality,
        metrics=("ndcg", "recall", "mrr"),
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "scored_target": out / "scored_target_runs.jsonl",
        "user_condition": out / "user_condition.jsonl",
        "identity_pairs": out / "identity_pairs.jsonl",
        "personality_pairs": out / "personality_pairs.jsonl",
        "inference": out / "inference.jsonl",
    }
    _write_jsonl(artifacts["scored_target"], combined)
    _write_jsonl(artifacts["user_condition"], aggregated)
    _write_jsonl(artifacts["identity_pairs"], identity)
    _write_jsonl(artifacts["personality_pairs"], personality)
    _write_jsonl(artifacts["inference"], inference)

    manifest = {
        "schema_version": "faireval-hosted-paired-analysis-v1",
        "scope": "synthetic_controlled_hosted_paired_extension",
        "real_world_claim_allowed": False,
        "target_users": sorted(target_users),
        "target_users_total": len(target_users),
        "target_cells": len(combined),
        "identity_pairs": len(identity),
        "personality_pairs": len(personality),
        "inference_rows": len(inference),
        "historical_pilot_reused_for_inference": False,
        "extension_run_sha256": file_sha256(Path(args.extension_run)),
        "extension_plan_manifest_sha256": file_sha256(
            Path(args.extension_plan_dir) / "plan_manifest.json"
        ),
        "artifacts": {
            name: file_sha256(path) for name, path in artifacts.items()
        },
    }
    manifest["manifest_sha256_without_self"] = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _run(
        "scripts/render_hosted_paired_results.py",
        "--inference-jsonl",
        str(artifacts["inference"]),
        "--user-condition-jsonl",
        str(artifacts["user_condition"]),
    )
    _run("scripts/check_paper_source.py")
    _run("scripts/check_result_table_contracts.py")
    _run(
        "scripts/build_overleaf_bundle.py",
        "--output",
        args.overleaf_zip,
    )
    _run("scripts/check_submission_closeout.py", "--skip-tests")

    print(json.dumps({
        "status": "PASS",
        "target_users": 9,
        "target_cells": 270,
        "identity_pairs": 54,
        "personality_pairs": 54,
        "analysis_manifest": str(out / "manifest.json"),
        "hosted_table": "paper/generated/fairsynth_hosted_table.tex",
        "hosted_summary": "paper/generated/fairsynth_hosted_summary.tex",
        "hosted_figure": "paper/figures/fairsynth_hosted_profiles.pdf",
        "overleaf_zip": args.overleaf_zip,
        "overleaf_zip_sha256": file_sha256(Path(args.overleaf_zip)),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
