from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from faireval.analysis import score_run_log
from faireval.freeze import file_sha256
from faireval.rq4_analysis import build_rq4_pair_artifact


def _sha256_json(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build validation-frozen contextual PAIR artifacts for FairEval RQ4."
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2027)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--utility-floor-ratio", type=float, default=0.95)
    args = parser.parse_args()

    scored = score_run_log(
        args.output_jsonl,
        plan_dir=args.plan_dir,
        freeze_root=args.freeze_root,
    )
    artifact = build_rq4_pair_artifact(
        scored,
        freeze_root=args.freeze_root,
        split_seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        utility_floor_ratio=args.utility_floor_ratio,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "rq4_pair_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "faireval-rq4-analysis-manifest-v1",
        "raw_run_log_sha256": file_sha256(args.output_jsonl),
        "run_plan_sha256": file_sha256(args.plan_dir / "run_plan.jsonl"),
        "plan_manifest_sha256": file_sha256(args.plan_dir / "plan_manifest.json"),
        "rq4_artifact_sha256": file_sha256(artifact_path),
        "rq4_artifact_content_sha256": _sha256_json(artifact),
        "selection_used_test_outcomes": False,
        "per_model_or_dataset_tuning": False,
    }
    manifest_path = args.output_dir / "rq4_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(artifact_path), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
