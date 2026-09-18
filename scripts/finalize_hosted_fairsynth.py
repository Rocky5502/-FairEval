from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from faireval.execute import completed_cell_ids, load_and_verify_plan
from faireval.freeze import file_sha256


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit hosted FairSynth results, run frozen inference, render the "
            "paper table, and rebuild the anonymous Overleaf bundle."
        )
    )
    parser.add_argument(
        "--plan-dir",
        default="results/plans/hosted-fairsynth-budget-v1",
    )
    parser.add_argument(
        "--freeze-root",
        default="data/frozen",
    )
    parser.add_argument(
        "--output-jsonl",
        default="results/runs/hosted-fairsynth-budget-v1.jsonl",
    )
    parser.add_argument(
        "--budget-ledger",
        default="results/budget/hosted_zzz_v1.json",
    )
    parser.add_argument(
        "--analysis-dir",
        default="results/analysis/fairsynth360-hosted-v1",
    )
    parser.add_argument(
        "--paper-table",
        default="paper/generated/fairsynth_hosted_table.tex",
    )
    parser.add_argument(
        "--paper-figure",
        default="paper/figures/fairsynth_hosted_effects.pdf",
    )
    parser.add_argument(
        "--overleaf-zip",
        default="dist/FairEval_ECIR2027_Overleaf_HOSTED_RESULTS.zip",
    )
    parser.add_argument(
        "--sync-manifest",
        default="results/paper_sync/hosted_fairsynth_v1.json",
    )
    args = parser.parse_args()

    plan_dir = Path(args.plan_dir)
    run_path = Path(args.output_jsonl)
    ledger_path = Path(args.budget_ledger)
    analysis_dir = Path(args.analysis_dir)
    paper_table = Path(args.paper_table)
    paper_figure = Path(args.paper_figure)
    bundle = Path(args.overleaf_zip)
    sync_path = Path(args.sync_manifest)

    if not run_path.is_file():
        raise FileNotFoundError(f"hosted run log not found: {run_path}")
    if not ledger_path.is_file():
        raise FileNotFoundError(f"budget ledger not found: {ledger_path}")

    cells, plan_manifest = load_and_verify_plan(plan_dir)
    completed = completed_cell_ids(run_path)
    if not completed:
        raise ValueError("hosted run log contains no persisted planned cells")

    _run(
        "scripts/audit_run_log.py",
        "--output-jsonl",
        str(run_path),
        "--plan-dir",
        str(plan_dir),
    )
    _run(
        "scripts/analyze_fairsynth360.py",
        "--output-jsonl",
        str(run_path),
        "--plan-dir",
        str(plan_dir),
        "--freeze-root",
        str(Path(args.freeze_root)),
        "--output-dir",
        str(analysis_dir),
    )

    inference_path = analysis_dir / "inference.jsonl"
    _run(
        "scripts/render_fairsynth_paper_results.py",
        "--inference-jsonl",
        str(inference_path),
        "--output",
        str(paper_table),
    )
    _run(
        "scripts/build_result_figures.py",
        "--fairsynth-inference",
        str(inference_path),
        "--output-dir",
        str(paper_figure.parent),
    )
    if not paper_figure.is_file():
        raise FileNotFoundError(f"hosted FairSynth paper figure not generated: {paper_figure}")
    _run("scripts/check_paper_source.py")
    _run("scripts/check_result_table_contracts.py")
    _run(
        "scripts/build_overleaf_bundle.py",
        "--output",
        str(bundle),
    )

    ledger = _load_json(ledger_path)
    latest = ledger.get("latest") if isinstance(ledger.get("latest"), dict) else {}
    analysis_manifest = _load_json(analysis_dir / "manifest.json")
    sync = {
        "schema_version": "faireval-hosted-fairsynth-paper-sync-v1",
        "plan_sha256": plan_manifest.get("plan_sha256"),
        "planned_cells": len(cells),
        "completed_cells": len(completed),
        "remaining_cells": len(cells) - len(completed),
        "complete_plan": len(completed) == len(cells),
        "budget_target_rmb": ledger.get("target_rmb"),
        "budget_hard_cap_rmb": ledger.get("hard_cap_rmb"),
        "spent_rmb_latest": latest.get("spent_rmb"),
        "run_jsonl_sha256": file_sha256(run_path),
        "budget_ledger_sha256": file_sha256(ledger_path),
        "analysis_manifest_sha256": file_sha256(analysis_dir / "manifest.json"),
        "inference_sha256": file_sha256(inference_path),
        "paper_table_sha256": file_sha256(paper_table),
        "paper_figure_sha256": file_sha256(paper_figure),
        "overleaf_zip_sha256": file_sha256(bundle),
        "analysis_scope": analysis_manifest.get("scope"),
        "real_world_claim_allowed": analysis_manifest.get("real_world_claim_allowed"),
        "paper_update_policy": "artifact_only_no_manual_numeric_transcription",
    }
    sync["sync_sha256_without_self"] = hashlib.sha256(
        json.dumps(sync, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    sync_path.write_text(
        json.dumps(sync, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sync, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
