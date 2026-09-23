from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from faireval.execute import load_and_verify_plan
from faireval.freeze import file_sha256, load_frozen_instances
from faireval.local_run_audit import audit_local_run_log
from faireval.output_protocol import analyze_ranking_output
from faireval.preexecution import load_preexecution_seal, verify_preexecution_seal
from faireval.prompts import candidate_selection_map


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOTAL = 2880
EXPECTED_PER_FAMILY = 1440
EXPECTED_REVISIONS = {
    "qwen25_local": "a09a35458c702b33eeacc393d103063234e8bc28",
    "phi35_local": "2fe192450127e6a83f7441aef6e3ca586c338b77",
}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the complete 2,880-cell V7 lean local run."
    )
    parser.add_argument("--plan-dir", default="results/plans/whitebox-v7-lean/core")
    parser.add_argument("--run-dir", default="results/runs/whitebox-v7-lean")
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument(
        "--preexecution-seal",
        default="results/preexecution/seal-v13/PREEXECUTION_SEAL.json",
    )
    parser.add_argument("--output-dir", default="results/analysis/whitebox-v7-lean")
    args = parser.parse_args()

    plan_dir = Path(args.plan_dir)
    run_dir = Path(args.run_dir)
    seal_path = Path(args.preexecution_seal)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cells, manifest = load_and_verify_plan(plan_dir)
    if manifest.get("schema_version") != "faireval-whitebox-v7-lean-plan-v1":
        raise ValueError("unexpected V7 lean plan schema")
    plan_by_id = {str(row["cell_id"]): row for row in cells}

    head = _git_head()
    failures: list[str] = []
    checks: dict[str, Any] = {}

    try:
        seal_verification = verify_preexecution_seal(
            seal_path,
            expected_commit_sha=head,
            plan_dir=plan_dir,
            plan_key="whitebox_v7_lean",
        )
        checks["preexecution_seal_pass"] = True
        checks["preexecution_seal_verification"] = seal_verification
    except Exception as exc:
        checks["preexecution_seal_pass"] = False
        checks["preexecution_seal_error"] = str(exc)
        failures.append("preexecution_seal")

    seal = load_preexecution_seal(seal_path)
    environment = seal.get("environment_check")
    checks["hardware_environment_pass"] = bool(
        isinstance(environment, dict) and environment.get("status") == "PASS"
    )
    if not checks["hardware_environment_pass"]:
        failures.append("hardware_environment")

    instances = {
        ("fairsynth360", str(instance.user_id)): instance
        for instance in load_frozen_instances(Path(args.freeze_root) / "fairsynth360")
    }

    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    family_integrity_audits: dict[str, Any] = {}
    for family in EXPECTED_REVISIONS:
        path = run_dir / f"{family}.jsonl"
        frows = _read_jsonl(path)
        if path.is_file():
            source_hashes[str(path)] = file_sha256(path)
        rows.extend(frows)
        try:
            family_integrity_audits[family] = audit_local_run_log(path, plan_dir=plan_dir)
        except Exception as exc:
            family_integrity_audits[family] = {"status": "FAIL", "error": str(exc)}
            failures.append(f"run_audit:{family}")

    cell_ids = [str(row.get("planned_cell_id", "")) for row in rows]
    duplicate_ids = len(cell_ids) - len(set(cell_ids))
    unknown_ids = sorted(set(cell_ids) - set(plan_by_id))
    missing_ids = sorted(set(plan_by_id) - set(cell_ids))
    checks.update({
        "row_count": len(rows),
        "expected_row_count": EXPECTED_TOTAL,
        "duplicate_planned_cell_ids": duplicate_ids,
        "unknown_planned_cell_ids": len(unknown_ids),
        "missing_planned_cell_ids": len(missing_ids),
    })
    if len(rows) != EXPECTED_TOTAL:
        failures.append("incomplete_main_run")
    if duplicate_ids:
        failures.append("duplicate_planned_cell_ids")
    if unknown_ids:
        failures.append("unknown_planned_cell_ids")
    if missing_ids:
        failures.append("missing_planned_cell_ids")

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    protocol_mismatch = 0
    mutation_count = 0
    ambiguity_count = 0
    commit_mismatch = 0
    model_revision_mismatch = 0
    hardware_mismatch = 0
    mapping_mismatch = 0

    for row in rows:
        family = str(row.get("model_family", ""))
        by_family[family].append(row)
        cell_id = str(row.get("planned_cell_id", ""))
        planned = plan_by_id.get(cell_id)
        if planned is None:
            continue
        instance = instances.get((str(planned["dataset"]), str(planned["user_id"])))
        if instance is None:
            failures.append(f"missing_instance:{cell_id}")
            continue

        selection_map = candidate_selection_map(
            instance,
            None if planned.get("candidate_order_seed") is None else int(planned["candidate_order_seed"]),
        )
        recalculated = analyze_ranking_output(
            str(row.get("raw_response", "")),
            candidate_ids=tuple(selection_map),
            k=int(planned["k"]),
        )
        persisted = row.get("output_protocol")
        if not isinstance(persisted, dict) or persisted != recalculated.as_dict():
            protocol_mismatch += 1

        expected_selection = None if recalculated.ranking is None else list(recalculated.ranking)
        if row.get("ranking_selection_ids") != expected_selection:
            mapping_mismatch += 1
        expected_decoded = (
            None if recalculated.ranking is None
            else [selection_map[value] for value in recalculated.ranking]
        )
        if row.get("ranking") != expected_decoded:
            mapping_mismatch += 1

        mutation_count += int(recalculated.candidate_id_mutation_detected)
        ambiguity_count += int(recalculated.parser_ambiguity)
        if str(row.get("code_commit_sha", "")) != head:
            commit_mismatch += 1

        expected_revision = EXPECTED_REVISIONS.get(family)
        metadata = row.get("provider_metadata")
        if not isinstance(metadata, dict):
            model_revision_mismatch += 1
            hardware_mismatch += 1
            continue
        observed_commit = metadata.get("model_commit_hash")
        if (
            expected_revision is None
            or str(row.get("resolved_model_version", "")) != expected_revision
            or str(metadata.get("model_revision_requested", "")) != expected_revision
            or (observed_commit not in (None, "") and str(observed_commit) != expected_revision)
        ):
            model_revision_mismatch += 1

        if isinstance(environment, dict):
            if (
                str(metadata.get("gpu_name", "")) != str(environment.get("gpu", ""))
                or str(metadata.get("cuda_version", "")) != str(environment.get("cuda_version", ""))
                or str(metadata.get("torch_version", "")) != str(environment.get("torch", ""))
            ):
                hardware_mismatch += 1

    checks.update({
        "protocol_recalculation_mismatches": protocol_mismatch,
        "selection_mapping_mismatches": mapping_mismatch,
        "candidate_id_mutation_rows": mutation_count,
        "parser_ambiguity_rows": ambiguity_count,
        "commit_mismatch_rows": commit_mismatch,
        "model_revision_mismatch_rows": model_revision_mismatch,
        "hardware_mismatch_rows": hardware_mismatch,
    })
    for label, value in (
        ("protocol_recalculation_mismatch", protocol_mismatch),
        ("selection_mapping_mismatch", mapping_mismatch),
        ("candidate_id_mutation", mutation_count),
        ("parser_ambiguity", ambiguity_count),
        ("commit_provenance", commit_mismatch),
        ("model_revision", model_revision_mismatch),
        ("hardware_provenance", hardware_mismatch),
    ):
        if value:
            failures.append(label)

    family_reports = {}
    for family in EXPECTED_REVISIONS:
        frows = by_family.get(family, [])
        total = len(frows)
        semantic_valid = sum(bool(row.get("semantic_ranking_valid")) for row in frows)
        strict_valid = sum(bool(row.get("strict_format_valid")) for row in frows)
        format_flags = Counter()
        semantic_flags = Counter()
        for row in frows:
            format_flags.update(str(x) for x in row.get("format_violations", []))
            semantic_flags.update(str(x) for x in row.get("semantic_errors", []))
        if total != EXPECTED_PER_FAMILY:
            failures.append(f"family_row_count:{family}")
        family_reports[family] = {
            "rows": total,
            "expected_rows": EXPECTED_PER_FAMILY,
            "semantic_valid_count": semantic_valid,
            "semantic_valid_rate": _rate(semantic_valid, total),
            "strict_format_valid_count": strict_valid,
            "strict_format_valid_rate": _rate(strict_valid, total),
            "invalid_output_count": total - semantic_valid,
            "format_violation_counts": dict(sorted(format_flags.items())),
            "semantic_error_counts": dict(sorted(semantic_flags.items())),
        }

    failures = sorted(set(failures))
    report = {
        "schema_version": "faireval-whitebox-v7-lean-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "analysis_allowed": not failures,
        "git_commit_sha": head,
        "plan_sha256": manifest.get("plan_sha256"),
        "run_plan_file_sha256": manifest.get("run_plan_file_sha256"),
        "source_run_sha256": source_hashes,
        "family_integrity_audits": family_integrity_audits,
        "checks": checks,
        "families": family_reports,
        "failures": failures,
        "interpretation": (
            "Semantic invalid outputs are measured system behavior and do not by themselves "
            "fail the completed main-run audit; they receive zero primary utility and contribute to IOD."
        ),
    }
    report_path = output_dir / "main_run_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
