from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> dict[str, object]:
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
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a FairEval checkout without making provider API calls"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="skip pytest; useful before dependencies are fully installed",
    )
    args = parser.parse_args()

    # Local workspaces are intentionally untracked by git.
    for relative in (
        Path(args.data_root) / "raw",
        Path(args.data_root) / "processed",
        Path(args.data_root) / "frozen",
        Path("results") / "raw",
        Path("results") / "plans",
        Path("results") / "analysis",
        Path("paper") / "generated",
    ):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)

    commands = [
        [sys.executable, "scripts/bootstrap_data_layout.py", "--data-root", args.data_root],
        [sys.executable, "scripts/check_config_consistency.py"],
        [sys.executable, "scripts/check_pilot_readiness.py"],
        [sys.executable, "scripts/check_paper_source.py"],
    ]
    if not args.skip_tests:
        commands.append([sys.executable, "-m", "pytest", "-q"])

    results = [_run(command) for command in commands]
    failed = [row for row in results if int(row["returncode"]) != 0]
    report = {
        "schema_version": "faireval-repository-setup-v1",
        "api_calls_made": 0,
        "workspace_created": True,
        "results": results,
        "offline_setup_passed": not failed,
        "note": (
            "This command never downloads restricted datasets or calls model APIs. "
            "Pilot readiness remains blocked until exact dataset releases, licenses, "
            "credentials/endpoints, and the Llama serving stack are frozen."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
