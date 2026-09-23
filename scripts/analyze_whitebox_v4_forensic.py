from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from faireval.execute import load_and_verify_plan
from faireval.freeze import canonical_json, file_sha256, load_frozen_instances
from faireval.output_protocol import PROTOCOL_VERSION, analyze_ranking_output


V4_FROZEN_COMMIT = "05d90102a3de2c90310eee76b75757d745d6c664"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc forensic salvage of frozen FairEval V4 raw first responses. "
            "No generative repair is used and original run artifacts are never modified."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        default=None,
        help="Repeat for each frozen V4 family JSONL.",
    )
    parser.add_argument("--plan-dir", default="results/plans/whitebox-full-v1/core")
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument(
        "--output-dir",
        default="results/analysis/whitebox-v4-forensic-v1",
    )
    args = parser.parse_args()

    inputs = (
        [Path(value) for value in args.input_jsonl]
        if args.input_jsonl
        else [
            Path("results/runs/whitebox-full-v4/qwen25_local.jsonl"),
            Path("results/runs/whitebox-full-v4/phi35_local.jsonl"),
        ]
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    plan_dir = Path(args.plan_dir)
    plan_rows, plan_manifest = load_and_verify_plan(plan_dir)
    plan_by_id = {str(row["cell_id"]): row for row in plan_rows}
    if len(plan_by_id) != len(plan_rows):
        raise ValueError("V4 plan contains duplicate cell IDs")

    datasets = sorted({str(row["dataset"]) for row in plan_rows})
    instance_by_key = {}
    for dataset in datasets:
        for instance in load_frozen_instances(Path(args.freeze_root) / dataset):
            instance_by_key[(dataset, str(instance.user_id))] = instance

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "forensic_rows.jsonl"
    summary_path = output_dir / "forensic_summary.json"
    manifest_path = output_dir / "manifest.json"

    forensic_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    seen_cells: set[str] = set()

    for source in inputs:
        source_hashes[str(source)] = file_sha256(source)
        for line_no, row in enumerate(_read_jsonl(source), start=1):
            if row.get("schema_version") != "faireval-run-v4":
                raise ValueError(f"{source}:{line_no}: forensic source must be faireval-run-v4")
            if row.get("code_commit_sha") != V4_FROZEN_COMMIT:
                raise ValueError(
                    f"{source}:{line_no}: V4 commit drift: {row.get('code_commit_sha')!r}"
                )
            cell_id = str(row.get("planned_cell_id", ""))
            if not cell_id or cell_id not in plan_by_id:
                raise ValueError(f"{source}:{line_no}: unknown planned_cell_id")
            if cell_id in seen_cells:
                raise ValueError(f"duplicate V4 planned_cell_id across forensic inputs: {cell_id}")
            seen_cells.add(cell_id)
            planned = plan_by_id[cell_id]
            key = (str(planned["dataset"]), str(planned["user_id"]))
            instance = instance_by_key.get(key)
            if instance is None:
                raise ValueError(f"{source}:{line_no}: missing frozen instance {key!r}")

            protocol = analyze_ranking_output(
                str(row.get("raw_response", "")),
                candidate_ids=instance.candidate_ids(),
                k=int(planned["k"]),
            )
            forensic_rows.append(
                {
                    "schema_version": "faireval-v4-forensic-row-v1",
                    "post_hoc": True,
                    "source_v4_commit": V4_FROZEN_COMMIT,
                    "source_file": str(source),
                    "source_line_no": line_no,
                    "planned_cell_id": cell_id,
                    "dataset": str(row.get("dataset")),
                    "user_id": str(row.get("user_id")),
                    "model_family": str(row.get("model_family")),
                    "condition_id": str(row.get("condition_id")),
                    "k": int(planned["k"]),
                    "source_initial_valid_v4": bool(row.get("initial_valid")),
                    "source_final_valid_v4": bool(row.get("final_valid")),
                    "source_had_generative_repair": row.get("repair") is not None,
                    "raw_response_sha256": str(row.get("response_sha256")),
                    "forensic_protocol": protocol.as_dict(),
                }
            )

    if len(seen_cells) != len(forensic_rows):
        raise AssertionError("forensic cell uniqueness invariant failed")

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in forensic_rows:
        by_family[row["model_family"]].append(row)

    family_summaries: dict[str, Any] = {}
    for family, rows in sorted(by_family.items()):
        total = len(rows)
        classes: Counter[str] = Counter()
        format_flags: Counter[str] = Counter()
        semantic_flags: Counter[str] = Counter()
        strict_valid = semantic_valid = recoverable_only = 0
        parser_ambiguity = mutation = markdown = 0
        for row in rows:
            protocol = row["forensic_protocol"]
            classes[str(protocol["forensic_class"])] += 1
            format_flags.update(str(x) for x in protocol["format_violations"])
            semantic_flags.update(str(x) for x in protocol["semantic_errors"])
            strict_valid += int(bool(protocol["strict_format_valid"]))
            semantic_valid += int(bool(protocol["semantic_ranking_valid"]))
            recoverable_only += int(
                bool(protocol["semantic_ranking_valid"])
                and bool(protocol["normalization_applied"])
            )
            parser_ambiguity += int(bool(protocol["parser_ambiguity"]))
            mutation += int(bool(protocol["candidate_id_mutation_detected"]))
            markdown += int("markdown_fence" in protocol["format_violations"])

        family_summaries[family] = {
            "rows": total,
            "strict_valid": {
                "count": strict_valid,
                "rate": _rate(strict_valid, total),
            },
            "deterministic_recoverable_semantic_valid": {
                "count": recoverable_only,
                "rate": _rate(recoverable_only, total),
            },
            "semantic_valid_after_deterministic_parsing": {
                "count": semantic_valid,
                "rate": _rate(semantic_valid, total),
            },
            "markdown_fence": {
                "count": markdown,
                "rate": _rate(markdown, total),
            },
            "wrong_k": {
                "count": semantic_flags["wrong_k"],
                "rate": _rate(semantic_flags["wrong_k"], total),
            },
            "out_of_candidate_item": {
                "count": semantic_flags["out_of_candidate_item"],
                "rate": _rate(semantic_flags["out_of_candidate_item"], total),
            },
            "duplicate_item_ids": {
                "count": semantic_flags["duplicate_item_ids"],
                "rate": _rate(semantic_flags["duplicate_item_ids"], total),
            },
            "wrong_top_level_schema": {
                "count": semantic_flags["wrong_top_level_schema"],
                "rate": _rate(semantic_flags["wrong_top_level_schema"], total),
            },
            "invalid_json_syntax": {
                "count": semantic_flags["invalid_json_syntax"],
                "rate": _rate(semantic_flags["invalid_json_syntax"], total),
            },
            "parser_ambiguity": {
                "count": parser_ambiguity,
                "rate": _rate(parser_ambiguity, total),
            },
            "candidate_id_mutation_detected": {
                "count": mutation,
                "rate": _rate(mutation, total),
            },
            "forensic_class_counts": dict(sorted(classes.items())),
            "format_violation_counts": dict(sorted(format_flags.items())),
            "semantic_error_counts": dict(sorted(semantic_flags.items())),
        }

    summary = {
        "schema_version": "faireval-v4-forensic-summary-v1",
        "post_hoc": True,
        "confirmatory": False,
        "purpose": "forensic_sensitivity_only",
        "source_v4_commit": V4_FROZEN_COMMIT,
        "parser_protocol_version": PROTOCOL_VERSION,
        "source_repairs_ignored": True,
        "generative_repair_used_in_forensic_analysis": False,
        "original_v4_artifacts_modified": False,
        "rows": len(forensic_rows),
        "families": family_summaries,
    }
    _write_jsonl(row_path, forensic_rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "faireval-v4-forensic-manifest-v1",
        "post_hoc": True,
        "confirmatory": False,
        "source_v4_commit": V4_FROZEN_COMMIT,
        "source_run_sha256": source_hashes,
        "plan_dir": str(plan_dir),
        "plan_sha256": plan_manifest.get("plan_sha256"),
        "run_plan_file_sha256": plan_manifest.get("run_plan_file_sha256"),
        "parser_policy": {
            "protocol_version": PROTOCOL_VERSION,
            "generative_repair": False,
            "source_repair_rows_ignored": True,
            "allowed_normalization": [
                "strip_surrounding_whitespace",
                "strip_single_recognized_json_fence",
                "extract_single_unambiguous_json_object",
            ],
            "forbidden_transformations": [
                "change_item_ids",
                "remove_leading_zeros",
                "truncate_ranking",
                "reorder_ranking",
                "replace_item_ids",
                "invent_item_ids",
            ],
        },
        "script_sha256": file_sha256(script_path),
        "artifacts": {
            "forensic_rows_sha256": file_sha256(row_path),
            "forensic_summary_sha256": file_sha256(summary_path),
        },
    }
    manifest["manifest_sha256_without_self"] = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "post_hoc": True,
                "output_dir": str(output_dir),
                "summary": summary,
                "manifest_sha256_without_self": manifest["manifest_sha256_without_self"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
