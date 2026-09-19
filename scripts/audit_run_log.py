from __future__ import annotations

import argparse
import json
from pathlib import Path

from faireval.run_audit import audit_run_log


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit FairEval raw JSONL provenance before statistical analysis"
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--plan-dir",
        help="optional immutable plan directory; when supplied, every run row is checked against it",
    )
    args = parser.parse_args()

    summary = audit_run_log(
        Path(args.output_jsonl),
        plan_dir=None if args.plan_dir is None else Path(args.plan_dir),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
