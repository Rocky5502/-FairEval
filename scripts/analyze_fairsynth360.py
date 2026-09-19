from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from faireval.analysis import aggregate_repetitions, score_run_log
from faireval.fairsynth_analysis import (
    build_fairsynth_identity_pairs,
    build_fairsynth_personality_pairs,
)
from faireval.freeze import canonical_json, file_sha256
from faireval.inference import summarize_paired_estimands


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze FairSynth-360 separately from real-world FairEval inference"
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument("--output-dir", default="results/analysis/fairsynth360-v1")
    args = parser.parse_args()

    scored = score_run_log(
        Path(args.output_jsonl),
        plan_dir=Path(args.plan_dir),
        freeze_root=Path(args.freeze_root),
    )
    scored = [row for row in scored if row["dataset"] == "fairsynth360"]
    if not scored:
        raise ValueError("run log contains no FairSynth-360 rows")

    aggregated = aggregate_repetitions(scored)
    identity = build_fairsynth_identity_pairs(aggregated)
    personality = build_fairsynth_personality_pairs(aggregated)
    inference = summarize_paired_estimands(
        identity + personality,
        metrics=("ndcg", "recall", "mrr"),
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "scored": out / "scored_runs.jsonl",
        "aggregated": out / "user_condition.jsonl",
        "identity_pairs": out / "identity_pairs.jsonl",
        "personality_pairs": out / "personality_pairs.jsonl",
        "inference": out / "inference.jsonl",
    }
    _write_jsonl(paths["scored"], scored)
    _write_jsonl(paths["aggregated"], aggregated)
    _write_jsonl(paths["identity_pairs"], identity)
    _write_jsonl(paths["personality_pairs"], personality)
    _write_jsonl(paths["inference"], inference)

    manifest = {
        "schema_version": "faireval-fairsynth-analysis-manifest-v1",
        "scope": "synthetic_controlled_stress_test_only",
        "real_world_claim_allowed": False,
        "input_run_sha256": file_sha256(Path(args.output_jsonl)),
        "artifacts": {name: file_sha256(path) for name, path in paths.items()},
        "counts": {
            "scored_runs": len(scored),
            "user_conditions": len(aggregated),
            "identity_pairs": len(identity),
            "personality_pairs": len(personality),
            "inference_rows": len(inference),
        },
    }
    manifest["manifest_sha256_without_self"] = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
