from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .execute import load_and_verify_plan
from .run_audit import audit_run_log
from .whitebox_analysis import LOCAL_FAMILIES


def audit_local_run_log(output_jsonl: Path, *, plan_dir: Path) -> dict[str, Any]:
    """Extend the general run audit with local deterministic-seed checks."""
    base = audit_run_log(output_jsonl, plan_dir=plan_dir)
    plan_rows, manifest = load_and_verify_plan(plan_dir)
    plan_by_id = {str(row["cell_id"]): row for row in plan_rows}

    checked = 0
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("model_family")) not in LOCAL_FAMILIES:
                continue
            cell_id = str(row.get("planned_cell_id", ""))
            planned = plan_by_id.get(cell_id)
            if planned is None:
                raise ValueError(f"line {line_no}: local cell not present in supplied plan")
            if planned.get("seed") is None:
                raise ValueError(f"line {line_no}: local plan cell is missing frozen seed")
            if row.get("seed_requested") != planned.get("seed"):
                raise ValueError(
                    f"line {line_no}: local requested seed drift: "
                    f"run={row.get('seed_requested')!r}, plan={planned.get('seed')!r}"
                )
            if row.get("seed_supported") is not True:
                raise ValueError(f"line {line_no}: local provider must support frozen generation seed")
            checked += 1

    if checked == 0:
        raise ValueError("run log contains no local open-weight rows to audit")
    return {
        **base,
        "schema_version": "faireval-local-run-audit-v1",
        "local_rows_seed_verified": checked,
        "local_plan_sha256": manifest.get("plan_sha256"),
    }
