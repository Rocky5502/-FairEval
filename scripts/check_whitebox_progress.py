from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


TARGETS = {
    "qwen25_local": 6480,
    "phi35_local": 6480,
}


def _read(path: Path) -> tuple[list[dict], list[tuple[int, str]]]:
    rows: list[dict] = []
    bad: list[tuple[int, str]] = []
    if not path.exists():
        return rows, bad
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                bad.append((line_no, str(exc)))
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                bad.append((line_no, "row is not a JSON object"))
    return rows, bad


def _recent_seconds(rows: list[dict]) -> float | None:
    times: list[datetime] = []
    for row in rows:
        raw = row.get("request_utc")
        if not raw:
            continue
        try:
            times.append(datetime.fromisoformat(str(raw)))
        except ValueError:
            continue
    intervals: list[float] = []
    for left, right in zip(times, times[1:]):
        seconds = (right - left).total_seconds()
        if 0 < seconds < 600:
            intervals.append(seconds)
    recent = intervals[-200:]
    return statistics.median(recent) if recent else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only progress audit for the canonical FairEval white-box campaign."
    )
    parser.add_argument(
        "--output-dir",
        default="results/runs/whitebox-full-v3",
    )
    parser.add_argument(
        "--expected-commit-sha",
        default=None,
    )
    args = parser.parse_args()

    root = Path(args.output_dir)
    total_target = sum(TARGETS.values())
    total_done = 0
    failed = False
    family_reports: dict[str, dict] = {}

    for family, target in TARGETS.items():
        path = root / f"{family}.jsonl"
        rows, bad = _read(path)
        ids = [
            str(row["planned_cell_id"])
            for row in rows
            if row.get("planned_cell_id")
        ]
        unique = set(ids)
        duplicate_count = len(ids) - len(unique)
        commits = sorted(
            {
                str(row["code_commit_sha"])
                for row in rows
                if row.get("code_commit_sha")
            }
        )
        seconds = _recent_seconds(rows)
        done = len(unique)
        remaining = max(0, target - done)
        total_done += done

        if bad or duplicate_count:
            failed = True
        if args.expected_commit_sha and commits and commits != [args.expected_commit_sha]:
            failed = True

        family_reports[family] = {
            "file": str(path),
            "file_exists": path.exists(),
            "valid_rows": len(rows),
            "unique_canonical_cells": done,
            "target_cells": target,
            "progress_percent": round(100.0 * done / target, 4),
            "remaining_cells": remaining,
            "duplicate_planned_cell_ids": duplicate_count,
            "bad_json_rows": len(bad),
            "code_commits": commits,
            "recent_median_seconds_per_cell": None if seconds is None else round(seconds, 3),
            "rough_eta_hours": (
                None if seconds is None else round(remaining * seconds / 3600.0, 2)
            ),
        }

    report = {
        "schema_version": "faireval-whitebox-progress-v1",
        "status": "FAIL" if failed else "PASS",
        "output_dir": str(root),
        "families": family_reports,
        "overall": {
            "completed_cells": total_done,
            "target_cells": total_target,
            "progress_percent": round(100.0 * total_done / total_target, 4),
            "remaining_cells": total_target - total_done,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
