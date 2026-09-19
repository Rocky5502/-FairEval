from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from faireval.analysis import score_run_log
from faireval.execute import load_and_verify_plan
from faireval.freeze import file_sha256
from faireval.rq3_analysis import build_factor_rows, summarize_factor_variation


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze factorized FairEval RQ3 robustness runs against the frozen core baseline"
    )
    parser.add_argument("--core-output-jsonl", required=True)
    parser.add_argument("--core-plan-dir", required=True)
    parser.add_argument("--rq3-output-jsonl", required=True)
    parser.add_argument("--rq3-plan-dir", required=True)
    parser.add_argument("--freeze-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-commit-sha", required=True)
    args = parser.parse_args()

    freeze_root = Path(args.freeze_root)
    core_raw = Path(args.core_output_jsonl)
    rq3_raw = Path(args.rq3_output_jsonl)
    core_plan_dir = Path(args.core_plan_dir)
    rq3_plan_dir = Path(args.rq3_plan_dir)
    output_dir = Path(args.output_dir)

    core_scored = score_run_log(core_raw, plan_dir=core_plan_dir, freeze_root=freeze_root)
    rq3_scored = score_run_log(rq3_raw, plan_dir=rq3_plan_dir, freeze_root=freeze_root)
    rq3_plan, rq3_manifest = load_and_verify_plan(rq3_plan_dir)
    plan_by_cell = {str(row["cell_id"]): row for row in rq3_plan}

    factor_rows = build_factor_rows(
        core_scored=core_scored,
        rq3_scored=rq3_scored,
        rq3_plan_by_cell=plan_by_cell,
    )
    summaries = summarize_factor_variation(factor_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    factor_path = output_dir / "rq3_factor_rows.jsonl"
    summary_path = output_dir / "rq3_variation_summary.jsonl"
    _write_jsonl(factor_path, factor_rows)
    _write_jsonl(summary_path, summaries)

    core_commits = {str(row["code_commit_sha"]) for row in core_scored}
    rq3_commits = {str(row["code_commit_sha"]) for row in rq3_scored}
    if len(core_commits) != 1 or len(rq3_commits) != 1:
        raise AssertionError("raw-run audit should have rejected mixed code commits")
    if core_commits != rq3_commits:
        raise ValueError(
            "RQ3 analysis requires the core and extra robustness runs to use the same code commit: "
            f"core={sorted(core_commits)}, rq3={sorted(rq3_commits)}"
        )

    manifest = {
        "schema_version": "faireval-rq3-analysis-manifest-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_commit_sha": args.analysis_commit_sha,
        "run_code_commit_sha": next(iter(core_commits)),
        "core_raw_sha256": file_sha256(core_raw),
        "rq3_raw_sha256": file_sha256(rq3_raw),
        "core_plan_manifest_sha256": file_sha256(core_plan_dir / "plan_manifest.json"),
        "rq3_plan_manifest_sha256": file_sha256(rq3_plan_dir / "plan_manifest.json"),
        "rq3_plan_sha256": rq3_manifest.get("plan_sha256"),
        "row_counts": {
            "factor_rows": len(factor_rows),
            "variation_summaries": len(summaries),
        },
        "artifact_sha256": {
            "factor_rows": file_sha256(factor_path),
            "variation_summary": file_sha256(summary_path),
        },
    }
    manifest_path = output_dir / "rq3_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
