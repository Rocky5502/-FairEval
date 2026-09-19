from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "ok": completed.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run FairEval Week-1 zero-provider-call bootstrap and write a machine-readable status report"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "bootstrap" / "week1_status.json",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    commands: list[tuple[str, list[str], bool]] = [
        ("config_consistency", [python, "scripts/check_config_consistency.py"], True),
        ("pilot_readiness_schema", [python, "scripts/check_pilot_readiness.py"], True),
        ("local_gpu_preflight", [python, "scripts/check_local_gpu.py"], True),
        (
            "fairsynth_oracle",
            [python, "scripts/run_fairsynth_oracle.py", "--output", "results/smoke/fairsynth_oracle_v1.json"],
            True,
        ),
        ("paper_figures", [python, "scripts/build_paper_figures.py"], True),
        ("paper_source", [python, "scripts/check_paper_source.py"], True),
        ("result_table_contracts", [python, "scripts/check_result_table_contracts.py"], True),
        (
            "environment_manifest",
            [python, "scripts/write_environment_manifest.py", "--output", "results/environment_manifest.json"],
            True,
        ),
    ]
    if not args.skip_tests:
        commands.insert(0, ("unit_tests", [python, "-m", "pytest", "-q"], True))

    results: dict[str, Any] = {}
    hard_failures: list[str] = []
    for name, command, required in commands:
        result = _run(command)
        results[name] = result
        print(f"[{name}] {'PASS' if result['ok'] else 'FAIL'}")
        if required and not result["ok"]:
            hard_failures.append(name)

    report = {
        "schema_version": "faireval-week1-bootstrap-v1",
        "python": sys.version,
        "repo_root": str(ROOT),
        "provider_calls_made": False,
        "steps": results,
        "hard_failures": hard_failures,
        "ready_for_next_local_step": not hard_failures,
        "note": (
            "This bootstrap intentionally performs no hosted API or local-model generation calls. "
            "Pilot-readiness may still report external dataset/credential/Llama-host blockers while returning zero in non-strict mode."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
