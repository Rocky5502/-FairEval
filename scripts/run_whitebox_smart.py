from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], *, label: str) -> None:
    print("\n" + "=" * 92, flush=True)
    print(label, flush=True)
    print("=" * 92, flush=True)
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(
            f"{label} failed with exit code {result.returncode}. "
            "Nothing later in the pipeline was started."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-command FairEval local white-box execution: environment gate, fresh "
            "scientific seal, one-cell-per-family canary, exact-resume full campaign, "
            "audit/finalize, and Overleaf rebuild."
        )
    )
    parser.add_argument(
        "--seal-dir",
        default="results/preexecution/seal-v7",
    )
    parser.add_argument(
        "--output-dir",
        default="results/runs/whitebox-full-v2",
    )
    parser.add_argument(
        "--max-cells-per-family",
        type=int,
        default=6480,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run GPU inference. Without this flag only the environment/seal/dry-run path executes.",
    )
    args = parser.parse_args()

    seal_dir = Path(args.seal_dir)
    seal_json = seal_dir / "PREEXECUTION_SEAL.json"
    output_dir = Path(args.output_dir)
    qwen_log = output_dir / "qwen25_local.jsonl"
    phi_log = output_dir / "phi35_local.jsonl"

    _run(
        ["scripts/check_whitebox_environment.py"],
        label="1/5 WHITE-BOX ENVIRONMENT GATE",
    )

    _run(
        ["scripts/build_preexecution_seal.py", "--output-dir", str(seal_dir)],
        label="2/5 FRESH PRE-EXECUTION SCIENTIFIC SEAL",
    )

    if not args.execute:
        _run(
            [
                "scripts/run_whitebox_full.py",
                "--preexecution-seal",
                str(seal_json),
                "--output-dir",
                str(output_dir),
                "--max-cells-per-family",
                str(args.max_cells_per_family),
            ],
            label="3/5 CANONICAL WHITE-BOX DRY RUN",
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "dry_run",
                    "seal": str(seal_json),
                    "next": "Re-run this same script with --execute.",
                },
                indent=2,
            )
        )
        return 0

    # Two real cells total: one Qwen + one Phi. This catches package/model/runtime
    # incompatibilities before committing hours to the full campaign. The exact same
    # output logs are reused below, so these are not throw-away generations.
    _run(
        [
            "scripts/run_whitebox_full.py",
            "--preexecution-seal",
            str(seal_json),
            "--output-dir",
            str(output_dir),
            "--max-cells-per-family",
            "1",
            "--execute",
        ],
        label="3/5 TWO-CELL REAL GPU CANARY (ONE CELL PER FAMILY)",
    )

    _run(
        [
            "scripts/run_whitebox_full.py",
            "--preexecution-seal",
            str(seal_json),
            "--output-dir",
            str(output_dir),
            "--max-cells-per-family",
            str(args.max_cells_per_family),
            "--execute",
        ],
        label="4/5 FULL CANONICAL CAMPAIGN WITH EXACT RESUME",
    )

    _run(
        [
            "scripts/finalize_whitebox.py",
            "--input-jsonl",
            str(qwen_log),
            "--input-jsonl",
            str(phi_log),
            "--plan-dir",
            "results/plans/whitebox-full-v1/core",
            "--freeze-root",
            "data/frozen",
        ],
        label="5/5 AUDIT + ANALYZE + PAPER TABLE + OVERLEAF REBUILD",
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": "execute",
                "seal": str(seal_json),
                "qwen_log": str(qwen_log),
                "phi_log": str(phi_log),
                "overleaf": "dist/FairEval_ECIR2027_Overleaf.zip",
                "manual_empirical_numbers_entered": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
