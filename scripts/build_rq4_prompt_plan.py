from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from faireval.execute import load_and_verify_plan
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

    source_manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_plan_sha = source_manifest_payload.get("plan_sha256")
    if not isinstance(source_plan_sha, str) or not source_plan_sha:
        raise ValueError("source plan manifest lacks plan_sha256")

    cells = build_identity_irrelevance_plan(_read_jsonl(source_plan))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "run_plan.jsonl"
    plan_path.write_text(
        "".join(canonical_json(row) + "\n" for row in cells),
        encoding="utf-8",
        newline="\n",
    )
    run_plan_file_sha256 = file_sha256(plan_path)
    semantic_plan_sha256 = hashlib.sha256(
        canonical_json(cells).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "faireval-run-plan-manifest-v1",
        "plan_track": "rq4_identity_irrelevance_prompting",
        "source_plan_sha256": source_plan_sha,
        "source_run_plan_file_sha256": file_sha256(source_plan),
        "source_plan_manifest_sha256": file_sha256(source_manifest),
        "prompt_mode": "identity_irrelevance",
        "planned_conditions": len(
            {
                (str(row["dataset"]), str(row["user_id"]), str(row["condition"]["condition_id"]))
                for row in cells
            }
        ),
        "planned_api_cells": len(cells),
        "run_plan_file_sha256": run_plan_file_sha256,
        "plan_sha256": semantic_plan_sha256,
        "selection_rule": "C1 plus confirmatory gender C2 from rq4_mitigation_baseline only",
    }
    manifest_path = out / "plan_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Prove the derived plan is consumable by the same executor/auditor as core.
    verified_cells, verified_manifest = load_and_verify_plan(out)
    if len(verified_cells) != len(cells) or verified_manifest["plan_sha256"] != semantic_plan_sha256:
        raise AssertionError("derived RQ4 prompting plan failed self-verification")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
