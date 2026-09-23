from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(label: str, args: list[str], *, allow_gate_failure: bool = False) -> int:
    print("\n" + "=" * 88, flush=True)
    print(f"FAIREVAL V5 RECOVERY STAGE 1: {label}", flush=True)
    print("=" * 88, flush=True)
    completed = subprocess.run([sys.executable, *args], cwd=ROOT)
    if completed.returncode != 0 and not allow_gate_failure:
        raise SystemExit(
            f"{label} failed with exit code {completed.returncode}; "
            "V5 canary scale-up has not been authorized."
        )
    return completed.returncode


def main() -> int:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if branch != "ecir-2027-v5-recovery":
        raise SystemExit(
            f"stage-1 recovery must run on ecir-2027-v5-recovery; current branch={branch!r}"
        )

    _run(
        "POST-HOC V4 FORENSIC SALVAGE",
        ["scripts/analyze_whitebox_v4_forensic.py"],
    )
    _run(
        "BUILD + VERIFY V5 CANARY SEAL",
        ["scripts/build_whitebox_v5_seal.py"],
    )
    _run(
        "EXECUTE SEALED 144-CELL V5 CANARY",
        ["scripts/run_whitebox_v5_canary.py", "--execute"],
    )
    gate_code = _run(
        "AUDIT PREDECLARED V5 PROMOTION GATE",
        ["scripts/audit_whitebox_v5_canary.py"],
        allow_gate_failure=True,
    )

    gate_path = ROOT / "results/analysis/whitebox-v5-canary/canary_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    summary = {
        "schema_version": "faireval-v5-recovery-stage1-v1",
        "git_commit_sha": head,
        "branch": branch,
        "v4_forensic": "results/analysis/whitebox-v4-forensic-v1/forensic_summary.json",
        "v5_seal": "results/preexecution/seal-v10/PREEXECUTION_SEAL.json",
        "v5_canary_runs": "results/runs/whitebox-v5-canary",
        "v5_gate": str(gate_path.relative_to(ROOT)),
        "promotion_allowed": bool(gate.get("promotion_allowed", False)),
        "gate_failures": gate.get("failures", []),
        "automatic_scale_up_performed": False,
        "next_rule": (
            "Do not run a full V5 campaign unless promotion_allowed is true and the "
            "audited gate is reviewed."
        ),
    }
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return gate_code


if __name__ == "__main__":
    raise SystemExit(main())
