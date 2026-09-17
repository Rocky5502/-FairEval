from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from faireval.execute import load_and_verify_plan
from faireval.freeze import canonical_json, file_sha256
from faireval.local_run_audit import audit_local_run_log
from faireval.whitebox_analysis import extract_whitebox_rows, summarize_whitebox_rows

try:  # package import under tests
    from scripts.build_overleaf_bundle import build_overleaf_bundle
    from scripts.render_whitebox_paper import render as render_whitebox_table
except ModuleNotFoundError:  # direct: python scripts/finalize_whitebox.py
    from build_overleaf_bundle import build_overleaf_bundle
    from render_whitebox_paper import render as render_whitebox_table


LOCAL_FAMILIES = {"qwen25_local", "phi35_local"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no run rows")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def merge_complete_local_logs(
    inputs: list[Path], *, plan_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit and merge family logs in immutable plan order.

    Paper rendering is deliberately gated on *complete* coverage of the supplied
    plan. Canary or partially resumed runs may be analyzed separately, but they
    cannot silently become the canonical white-box paper table through this
    finalizer.
    """
    if len(inputs) < 2:
        raise ValueError("white-box finalization requires both family logs")

    plan_rows, plan_manifest = load_and_verify_plan(plan_dir)
    plan_by_id = {str(row["cell_id"]): row for row in plan_rows}
    planned_ids = [str(row["cell_id"]) for row in plan_rows]
    expected_ids = set(planned_ids)
    expected_families = {str(row["model_family"]) for row in plan_rows}
    if expected_families != LOCAL_FAMILIES:
        raise ValueError(
            "canonical white-box finalizer expects exactly qwen25_local and phi35_local; "
            f"plan has {sorted(expected_families)}"
        )

    by_id: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    observed_families: set[str] = set()
    for path in inputs:
        audit_local_run_log(path, plan_dir=plan_dir)
        source_hashes[str(path)] = file_sha256(path)
        for row in _read_jsonl(path):
            cell_id = str(row.get("planned_cell_id", ""))
            if cell_id not in plan_by_id:
                raise ValueError(f"{path}: row references a cell outside the supplied plan")
            if cell_id in by_id:
                raise ValueError(f"duplicate planned_cell_id across local logs: {cell_id}")
            family = str(row.get("model_family", ""))
            if family not in LOCAL_FAMILIES:
                raise ValueError(f"unexpected local family in merged logs: {family!r}")
            if family != str(plan_by_id[cell_id]["model_family"]):
                raise ValueError(f"run family disagrees with plan for cell {cell_id}")
            observed_families.add(family)
            by_id[cell_id] = row

    missing = expected_ids - set(by_id)
    extra = set(by_id) - expected_ids
    if extra:
        raise ValueError(f"merged local logs contain {len(extra)} unexpected cells")
    if missing:
        counts: dict[str, int] = {}
        for cell_id in missing:
            family = str(plan_by_id[cell_id]["model_family"])
            counts[family] = counts.get(family, 0) + 1
        raise ValueError(
            "white-box paper finalization requires complete plan coverage; "
            f"missing={len(missing)} by_family={counts}"
        )
    if observed_families != LOCAL_FAMILIES:
        raise ValueError(f"merged logs do not contain both local families: {sorted(observed_families)}")

    merged = [by_id[cell_id] for cell_id in planned_ids]
    provenance = {
        "schema_version": "faireval-whitebox-merge-v1",
        "plan_sha256": plan_manifest.get("plan_sha256"),
        "run_plan_file_sha256": plan_manifest.get("run_plan_file_sha256"),
        "planned_cells": len(plan_rows),
        "merged_cells": len(merged),
        "families": sorted(observed_families),
        "input_sha256": source_hashes,
        "complete_plan_coverage": True,
    }
    return merged, provenance


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit, merge, analyze, render, and bundle a complete two-family "
            "FairEval white-box run. No empirical value is entered by hand."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        required=True,
        help="repeat for each local family run log",
    )
    parser.add_argument(
        "--plan-dir",
        default="results/plans/whitebox-full-v1/core",
    )
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument(
        "--output-dir",
        default="results/analysis/whitebox-full-v1",
    )
    parser.add_argument(
        "--paper-table",
        default="paper/generated/whitebox_summary_table.tex",
    )
    parser.add_argument(
        "--overleaf-output",
        default="dist/FairEval_ECIR2027_Overleaf.zip",
    )
    args = parser.parse_args()

    inputs = [Path(value) for value in args.input_jsonl]
    plan_dir = Path(args.plan_dir)
    freeze_root = Path(args.freeze_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged, merge_provenance = merge_complete_local_logs(inputs, plan_dir=plan_dir)
    merged_path = output_dir / "merged_run.jsonl"
    _write_jsonl(merged_path, merged)

    # Re-audit the merged artifact, then join to deterministic recommendation
    # utility and compute the pre-declared descriptive white-box summaries.
    merged_audit = audit_local_run_log(merged_path, plan_dir=plan_dir)
    whitebox_rows = extract_whitebox_rows(
        merged_path,
        plan_dir=plan_dir,
        freeze_root=freeze_root,
    )
    whitebox_summary = summarize_whitebox_rows(whitebox_rows)

    rows_path = output_dir / "whitebox_rows.jsonl"
    summary_path = output_dir / "whitebox_summary.jsonl"
    _write_jsonl(rows_path, whitebox_rows)
    _write_jsonl(summary_path, whitebox_summary)

    paper_table = Path(args.paper_table)
    paper_table.parent.mkdir(parents=True, exist_ok=True)
    paper_table.write_text(
        render_whitebox_table(whitebox_summary),
        encoding="utf-8",
        newline="\n",
    )

    manifest = {
        "schema_version": "faireval-whitebox-finalization-v1",
        "scope": "separate_local_open_weight_whitebox_stratum",
        "complete_plan_coverage": True,
        "calibrated_uncertainty_claimed": False,
        "confirmatory_whitebox_p_values": False,
        "merge": merge_provenance,
        "merged_audit": merged_audit,
        "artifacts": {
            "merged_run": file_sha256(merged_path),
            "whitebox_rows": file_sha256(rows_path),
            "whitebox_summary": file_sha256(summary_path),
            "paper_table": file_sha256(paper_table),
        },
        "counts": {
            "run_rows": len(merged),
            "whitebox_rows": len(whitebox_rows),
            "summary_rows": len(whitebox_summary),
        },
    }
    manifest["manifest_sha256_without_self"] = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    bundle = build_overleaf_bundle(
        output_zip=Path(args.overleaf_output),
        include_available_results=True,
    )
    result = {
        "status": "PASS",
        "manifest": str(manifest_path),
        "paper_table": str(paper_table),
        "overleaf_bundle": bundle,
        "empirical_numbers_manually_entered": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
