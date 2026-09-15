from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from faireval.analysis import (
    aggregate_repetitions,
    build_rq1_confirmatory_pairs,
    build_rq2_pairs,
    score_run_log,
)
from faireval.freeze import file_sha256
from faireval.inference import summarize_paired_estimands


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build audited FairEval RQ1/RQ2 analysis artifacts from a frozen run log"
    )
    parser.add_argument("--output-jsonl", required=True, help="raw faireval-run-v4 JSONL")
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--freeze-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-config", default="configs/analysis.yaml")
    parser.add_argument(
        "--analysis-commit-sha",
        help="analysis code commit; defaults to git rev-parse HEAD when available",
    )
    args = parser.parse_args()

    raw_path = Path(args.output_jsonl)
    plan_dir = Path(args.plan_dir)
    freeze_root = Path(args.freeze_root)
    output_dir = Path(args.output_dir)
    config_path = Path(args.analysis_config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    invalid_policy = config["invalid_outputs"]["primary_system_utility"]
    if invalid_policy != "zero":
        raise ValueError(
            "analysis code and frozen policy disagree: primary invalid-output utility must be zero"
        )
    inference = config["inference"]
    if inference["multiple_comparisons"]["method"] != "holm":
        raise ValueError("current frozen analysis supports Holm correction only")
    if inference["multiple_comparisons"]["family"] != "rq_metric_contrast":
        raise ValueError("current frozen Holm family must be rq_metric_contrast")

    scored = score_run_log(
        raw_path,
        plan_dir=plan_dir,
        freeze_root=freeze_root,
        invalid_utility_policy=invalid_policy,
    )
    aggregated = aggregate_repetitions(scored)
    rq1_pairs = build_rq1_confirmatory_pairs(aggregated)
    rq2_pairs = build_rq2_pairs(aggregated)
    all_pairs = rq1_pairs + rq2_pairs

    metrics = [config["utility"]["primary_metric"]] + list(
        config["utility"].get("additional_metrics", [])
    )
    permutation_cfg = inference["paired_permutation"]
    summaries = summarize_paired_estimands(
        all_pairs,
        metrics=metrics,
        bootstrap_samples=int(inference["bootstrap_samples"]),
        bootstrap_seed=int(inference["bootstrap_seed"]),
        permutation_exact_max_n=int(permutation_cfg["exact_max_nonzero_pairs"]),
        permutation_samples=int(permutation_cfg["monte_carlo_samples"]),
        permutation_seed=int(permutation_cfg["seed"]),
        confidence=float(inference["confidence"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "scored_runs": output_dir / "scored_runs.jsonl",
        "user_condition": output_dir / "user_condition.jsonl",
        "rq1_pairs": output_dir / "rq1_pairs.jsonl",
        "rq2_pairs": output_dir / "rq2_pairs.jsonl",
        "inference": output_dir / "inference.jsonl",
    }
    _write_jsonl(paths["scored_runs"], scored)
    _write_jsonl(paths["user_condition"], aggregated)
    _write_jsonl(paths["rq1_pairs"], rq1_pairs)
    _write_jsonl(paths["rq2_pairs"], rq2_pairs)
    _write_jsonl(paths["inference"], summaries)

    run_commits = sorted({str(row["code_commit_sha"]) for row in scored})
    if len(run_commits) != 1:
        raise AssertionError("run audit should have rejected mixed code commits")
    analysis_commit = args.analysis_commit_sha or _git_sha()
    if not analysis_commit:
        raise ValueError("analysis commit SHA is required when git metadata is unavailable")

    plan_manifest_path = plan_dir / "plan_manifest.json"
    manifest = {
        "schema_version": "faireval-analysis-manifest-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_commit_sha": analysis_commit,
        "run_code_commit_sha": run_commits[0],
        "raw_run_sha256": file_sha256(raw_path),
        "plan_manifest_sha256": file_sha256(plan_manifest_path),
        "analysis_config_sha256": file_sha256(config_path),
        "invalid_output_primary_utility_policy": invalid_policy,
        "row_counts": {
            "scored_runs": len(scored),
            "user_condition": len(aggregated),
            "rq1_pairs": len(rq1_pairs),
            "rq2_pairs": len(rq2_pairs),
            "inference": len(summaries),
        },
        "artifact_sha256": {name: file_sha256(path) for name, path in paths.items()},
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
