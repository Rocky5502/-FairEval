from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from faireval.freeze import canonical_json, file_sha256
from faireval.rq4_prompting import build_identity_irrelevance_plan


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive the immutable RQ4 identity-irrelevance prompting plan from a frozen audit plan"
    )
    parser.add_argument("--source-plan-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_plan_dir)
    source_plan = source_dir / "run_plan.jsonl"
    source_manifest = source_dir / "plan_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)

    cells = build_identity_irrelevance_plan(_read_jsonl(source_plan))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "run_plan.jsonl"
    plan_path.write_text(
        "".join(canonical_json(row) + "\n" for row in cells),
        encoding="utf-8",
        newline="\n",
    )
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "faireval-rq4-prompt-plan-manifest-v1",
        "source_plan_sha256": file_sha256(source_plan),
        "source_plan_manifest_sha256": file_sha256(source_manifest),
        "prompt_mode": "identity_irrelevance",
        "planned_cells": len(cells),
        "plan_sha256": plan_sha,
        "selection_rule": "C1 plus confirmatory gender C2 from rq4_mitigation_baseline only",
    }
    (out / "plan_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
