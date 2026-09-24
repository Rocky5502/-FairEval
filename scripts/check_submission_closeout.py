from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "configs" / "submission_scope_ecir2027.yaml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one YAML mapping")
    return payload


def _run(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Close out the ECIR 2027 submission scope without faking deferred dataset freezes."
    )
    parser.add_argument(
        "--output",
        default="results/submission/ecir2027_closeout.json",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the full pytest suite while still running paper/config preflights.",
    )
    args = parser.parse_args()

    scope = _load_yaml(SCOPE)
    failures: list[str] = []

    if scope.get("status") != "submission_scope_frozen":
        failures.append("submission scope is not frozen")

    completed = scope.get("completed_empirical_evidence", {})
    local = completed.get("local_v7_fairsynth", {})
    hosted = completed.get("hosted_fairsynth_operational_pilot", {})

    if local.get("status") != "complete" or int(local.get("observed_cells", -1)) != 2880:
        failures.append("local V7 completion contract is not 2,880/2,880")
    if local.get("additional_gpu_inference_required") is not False:
        failures.append("submission scope still requires GPU inference")
    if hosted.get("status") != "complete_operational_only":
        failures.append("hosted pilot is not frozen as operational-only evidence")
    if hosted.get("scientific_effect_inference_allowed") is not False:
        failures.append("hosted 99-cell pilot incorrectly permits scientific effect inference")

    # Regenerate the derived hosted paper artifacts before validating file presence,
    # so the closeout command is self-contained on a clean checkout.
    hosted_render = _run([sys.executable, "scripts/render_hosted_pilot_operational.py"])
    if not hosted_render["ok"]:
        failures.append(
            f"hosted_pilot_artifacts failed with return code {hosted_render['returncode']}"
        )

    required_files = [ROOT / p for p in scope.get("paper_artifacts_required", [])]
    missing = [str(p.relative_to(ROOT)) for p in required_files if not p.is_file()]
    if missing:
        failures.append(f"missing paper artifacts: {missing}")

    secondary_paths = (
        ROOT / "results" / "analysis" / "fairsynth-v7-secondary" / "secondary_summary.json",
        ROOT / "paper" / "figures" / "v7_secondary_profiles.pdf",
        ROOT / "paper" / "generated" / "v7_repetition_stability_table.tex",
    )
    secondary_present = [path.is_file() for path in secondary_paths]
    if any(secondary_present):
        if not all(secondary_present):
            failures.append(
                "secondary V7 artifacts are partially present; expected summary, figure, and stability table"
            )
        else:
            zero_byte = [
                str(path.relative_to(ROOT))
                for path in secondary_paths
                if path.stat().st_size <= 0
            ]
            if zero_byte:
                failures.append(f"secondary V7 artifacts contain zero-byte files: {zero_byte}")
            else:
                try:
                    secondary = json.loads(secondary_paths[0].read_text(encoding="utf-8"))
                    if secondary.get("schema_version") != "faireval-v7-secondary-analysis-v1":
                        failures.append("secondary V7 summary schema mismatch")
                    if secondary.get("new_model_or_api_calls") is not False:
                        failures.append("secondary V7 summary does not certify zero new model/API calls")
                    stability = secondary.get("repetition_stability_summary", [])
                    if len(stability) != 2:
                        failures.append("secondary V7 repetition stability must contain exactly two model rows")
                except (OSError, json.JSONDecodeError) as exc:
                    failures.append(f"secondary V7 summary unreadable: {exc}")

    provenance = ROOT / "provenance" / "V7_FINAL_RESULTS_2026-09-24.md"
    if not provenance.is_file():
        failures.append("V7 final-results provenance is missing")
    else:
        ptxt = provenance.read_text(encoding="utf-8")
        for token in (
            str(local.get("execution_commit")),
            str(local.get("plan_sha256")),
            str(local.get("preexecution_seal_sha256")),
            str(local.get("merged_run_sha256")),
            str(local.get("inference_sha256")),
            "2,880 planned and observed cells",
            "99 cells",
        ):
            if token not in ptxt:
                failures.append(f"provenance missing frozen token: {token}")

    main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    required_paper_markers = (
        "complete 2,880-generation local matrix",
        "operational feasibility, not fairness inference",
        "does not show that one hosted family is fairer than another",
        "registered real-world demographic/personality comparisons and mitigation extension still require",
    )
    for marker in required_paper_markers:
        if marker not in main:
            failures.append(f"paper scope guard missing: {marker}")

    forbidden_claims = (
        "hosted models demonstrate significant fairness",
        "we find that one hosted model is fairer",
        "real-world results show",
    )
    lowered = main.lower()
    for phrase in forbidden_claims:
        if phrase in lowered:
            failures.append(f"forbidden unsupported claim present: {phrase}")

    commands: dict[str, dict[str, Any]] = {
        "hosted_pilot_artifacts": hosted_render,
    }
    checks = [
        ("config_consistency", [sys.executable, "scripts/check_config_consistency.py"]),
        ("paper_source", [sys.executable, "scripts/check_paper_source.py"]),
        ("result_tables", [sys.executable, "scripts/check_result_table_contracts.py"]),
        ("overleaf_bundle", [
            sys.executable,
            "scripts/build_overleaf_bundle.py",
            "--output",
            "dist/FairEval_ECIR2027_Overleaf.zip",
        ]),
    ]
    if not args.skip_tests:
        checks.insert(0, ("pytest", [sys.executable, "-m", "pytest", "-q"]))

    for name, command in checks:
        result = _run(command)
        commands[name] = result
        if not result["ok"]:
            failures.append(f"{name} failed with return code {result['returncode']}")

    deferred = scope.get("deferred_not_claimed_in_submission", {})
    deferred_status = {
        key: {
            "status": value.get("status"),
            "required_for_current_submission_closeout": value.get(
                "required_for_current_submission_closeout"
            ),
        }
        for key, value in deferred.items()
        if isinstance(value, dict)
    }

    bundle = ROOT / "dist" / "FairEval_ECIR2027_Overleaf.zip"
    report = {
        "schema_version": "faireval-ecir2027-submission-closeout-v1",
        "status": "PASS" if not failures else "FAIL",
        "submission_scope": str(SCOPE.relative_to(ROOT)),
        "local_gpu_work_complete": local.get("additional_gpu_inference_required") is False,
        "current_submission_requires_real_world_dataset_freezes": False,
        "strict_paid_pilot_readiness_expected_to_remain_false": True,
        "deferred_not_claimed": deferred_status,
        "paper_artifacts_present": not missing,
        "commands": commands,
        "overleaf_zip": str(bundle.relative_to(ROOT)) if bundle.is_file() else None,
        "overleaf_zip_sha256": _sha256(bundle) if bundle.is_file() else None,
        "failures": failures,
    }

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
