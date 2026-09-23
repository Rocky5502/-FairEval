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
from faireval.prompts import candidate_selection_map
from faireval.preexecution import load_preexecution_seal, verify_preexecution_seal


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOTAL = 144
EXPECTED_PER_FAMILY = 72
EXPECTED_REVISIONS = {
    "qwen25_local": "a09a35458c702b33eeacc393d103063234e8bc28",
    "phi35_local": "2fe192450127e6a83f7441aef6e3ca586c338b77",
}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        description="Audit the complete 144-cell V7 canary against the predeclared promotion gate."
    )
    parser.add_argument("--plan-dir", default="results/plans/whitebox-v7-canary/core")
    parser.add_argument("--run-dir", default="results/runs/whitebox-v7-canary")
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument(
        "--preexecution-seal",
        default="results/preexecution/seal-v12/PREEXECUTION_SEAL.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis/whitebox-v7-canary",
    )
    args = parser.parse_args()

    plan_dir = Path(args.plan_dir)
    run_dir = Path(args.run_dir)
    seal_path = Path(args.preexecution_seal)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "canary_gate.json"

    cells, manifest = load_and_verify_plan(plan_dir)
    plan_by_id = {str(row["cell_id"]): row for row in cells}
    gate = manifest.get("promotion_gate")
    if not isinstance(gate, dict):
        raise ValueError("V7 canary plan lacks predeclared promotion gate")
    threshold = float(gate["semantic_exact_k_candidate_valid_rate_per_model_min"])
    if threshold != 0.95:
        raise ValueError(f"unexpected V7 canary promotion threshold {threshold}")

    head = _git_head()
    failures: list[str] = []
    checks: dict[str, Any] = {}

    try:
        seal_verification = verify_preexecution_seal(
            seal_path,
            expected_commit_sha=head,
            plan_dir=plan_dir,
            plan_key="whitebox_v7_canary",
        )
        checks["preexecution_seal_pass"] = True
        checks["preexecution_seal_verification"] = seal_verification
    except Exception as exc:
        checks["preexecution_seal_pass"] = False
        checks["preexecution_seal_error"] = str(exc)
        failures.append("preexecution_seal")

    seal = load_preexecution_seal(seal_path)
    environment = seal.get("environment_check")
    if not isinstance(environment, dict) or environment.get("status") != "PASS":
        checks["hardware_environment_pass"] = False
        failures.append("hardware_environment")
    else:
        checks["hardware_environment_pass"] = True

    instances = {}
    for instance in load_frozen_instances(Path(args.freeze_root) / "fairsynth360"):
        instances[("fairsynth360", str(instance.user_id))] = instance

    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    family_integrity_audits: dict[str, Any] = {}
    for family in EXPECTED_REVISIONS:
        path = run_dir / f"{family}.jsonl"
        family_rows = _read_jsonl(path)
        if path.is_file():
            source_hashes[str(path)] = file_sha256(path)
        rows.extend(family_rows)
        try:
            family_integrity_audits[family] = audit_local_run_log(path, plan_dir=plan_dir)
        except Exception as exc:
            family_integrity_audits[family] = {"status": "FAIL", "error": str(exc)}
            failures.append(f"run_audit:{family}")

    cell_ids = [str(row.get("planned_cell_id", "")) for row in rows]
    duplicate_ids = len(cell_ids) - len(set(cell_ids))
    unknown_ids = sorted(set(cell_ids) - set(plan_by_id))
    missing_ids = sorted(set(plan_by_id) - set(cell_ids))
    checks["row_count"] = len(rows)
    checks["expected_row_count"] = EXPECTED_TOTAL
    checks["duplicate_planned_cell_ids"] = duplicate_ids
    checks["unknown_planned_cell_ids"] = len(unknown_ids)
    checks["missing_planned_cell_ids"] = len(missing_ids)
    if len(rows) != EXPECTED_TOTAL:
        failures.append("incomplete_canary")
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

    for index, row in enumerate(rows, start=1):
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
        persisted_selection = row.get("ranking_selection_ids")
        expected_selection = None if recalculated.ranking is None else list(recalculated.ranking)
        if persisted_selection != expected_selection:
            protocol_mismatch += 1
        expected_decoded = (
            None
            if recalculated.ranking is None
            else [selection_map[value] for value in recalculated.ranking]
        )
        if row.get("ranking") != expected_decoded:
            protocol_mismatch += 1

        mutation_count += int(recalculated.candidate_id_mutation_detected)
        ambiguity_count += int(recalculated.parser_ambiguity)
        if str(row.get("code_commit_sha", "")) != head:
            commit_mismatch += 1

        expected_revision = EXPECTED_REVISIONS.get(family)
        provider_metadata = row.get("provider_metadata")
        if not isinstance(provider_metadata, dict):
            model_revision_mismatch += 1
            hardware_mismatch += 1
            continue
        observed_commit_hash = provider_metadata.get("model_commit_hash")
        if (
            expected_revision is None
            or str(row.get("resolved_model_version", "")) != expected_revision
            or str(provider_metadata.get("model_revision_requested", "")) != expected_revision
            or (
                observed_commit_hash not in (None, "")
                and str(observed_commit_hash) != expected_revision
            )
        ):
            model_revision_mismatch += 1

        if isinstance(environment, dict):
            if (
                str(provider_metadata.get("gpu_name", "")) != str(environment.get("gpu", ""))
                or str(provider_metadata.get("cuda_version", "")) != str(environment.get("cuda_version", ""))
                or str(provider_metadata.get("torch_version", "")) != str(environment.get("torch", ""))
            ):
                hardware_mismatch += 1

    checks["protocol_recalculation_mismatches"] = protocol_mismatch
    checks["candidate_id_mutation_rows"] = mutation_count
    checks["parser_ambiguity_rows"] = ambiguity_count
    checks["commit_mismatch_rows"] = commit_mismatch
    checks["model_revision_mismatch_rows"] = model_revision_mismatch
    checks["hardware_mismatch_rows"] = hardware_mismatch

    if protocol_mismatch:
        failures.append("protocol_recalculation_mismatch")
    if mutation_count:
        failures.append("candidate_id_mutation")
    if ambiguity_count:
        failures.append("parser_ambiguity")
    if commit_mismatch:
        failures.append("commit_provenance")
    if model_revision_mismatch:
        failures.append("model_revision")
    if hardware_mismatch:
        failures.append("hardware_provenance")

    family_reports: dict[str, Any] = {}
    for family, expected_revision in EXPECTED_REVISIONS.items():
        frows = by_family.get(family, [])
        total = len(frows)
        semantic_valid = sum(bool(row.get("semantic_ranking_valid")) for row in frows)
        strict_valid = sum(bool(row.get("strict_format_valid")) for row in frows)
        normalized = sum(bool(row.get("deterministic_normalization_applied")) for row in frows)
        format_flags: Counter[str] = Counter()
        semantic_flags: Counter[str] = Counter()
        for row in frows:
            format_flags.update(str(x) for x in row.get("format_violations", []))
            semantic_flags.update(str(x) for x in row.get("semantic_errors", []))
        semantic_rate = _rate(semantic_valid, total)
        family_pass = total == EXPECTED_PER_FAMILY and semantic_rate >= threshold
        if not family_pass:
            failures.append(f"semantic_validity_gate:{family}")
        family_reports[family] = {
            "rows": total,
            "expected_rows": EXPECTED_PER_FAMILY,
            "expected_model_revision": expected_revision,
            "semantic_valid_count": semantic_valid,
            "semantic_valid_rate": semantic_rate,
            "strict_format_valid_count": strict_valid,
            "strict_format_valid_rate": _rate(strict_valid, total),
            "deterministically_normalized_count": normalized,
            "deterministically_normalized_rate": _rate(normalized, total),
            "threshold": threshold,
            "promotion_subgate_pass": family_pass,
            "format_violation_counts": dict(sorted(format_flags.items())),
            "semantic_error_counts": dict(sorted(semantic_flags.items())),
        }

    failures = sorted(set(failures))
    promotion_allowed = not failures
    report = {
        "schema_version": "faireval-whitebox-v7-canary-gate-v1",
        "status": "PASS" if promotion_allowed else "FAIL",
        "promotion_allowed": promotion_allowed,
        "predeclared_gate": gate,
        "git_commit_sha": head,
        "plan_sha256": manifest.get("plan_sha256"),
        "run_plan_file_sha256": manifest.get("run_plan_file_sha256"),
        "source_run_sha256": source_hashes,
        "family_integrity_audits": family_integrity_audits,
        "checks": checks,
        "families": family_reports,
        "failures": failures,
        "decision_rule": (
            "Scale only when every hard check passes and each model has semantic "
            "exact-k candidate validity >= 0.95 over all 72 sealed canary cells."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if promotion_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
