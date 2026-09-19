from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from faireval.freeze import canonical_json, file_sha256
from faireval.whitebox_analysis import extract_whitebox_rows, summarize_whitebox_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze auxiliary token-level diagnostics from local open-weight FairEval runs"
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--freeze-root", default="data/frozen")
    parser.add_argument("--output-dir", default="results/analysis/local-whitebox-v1")
    args = parser.parse_args()

    rows = extract_whitebox_rows(
        Path(args.output_jsonl),
        plan_dir=Path(args.plan_dir),
        freeze_root=Path(args.freeze_root),
    )
    summaries = summarize_whitebox_rows(rows)

    out = Path(args.output_dir)
    row_path = out / "whitebox_rows.jsonl"
    summary_path = out / "whitebox_summary.jsonl"
    _write_jsonl(row_path, rows)
    _write_jsonl(summary_path, summaries)

    manifest = {
        "schema_version": "faireval-local-whitebox-analysis-manifest-v1",
        "scope": "exploratory_local_open_weight_only",
        "calibrated_uncertainty_claimed": False,
        "confirmatory_p_values": False,
        "input_run_sha256": file_sha256(Path(args.output_jsonl)),
        "artifacts": {
            "whitebox_rows": file_sha256(row_path),
            "whitebox_summary": file_sha256(summary_path),
        },
        "counts": {
            "rows": len(rows),
            "summary_rows": len(summaries),
        },
    }
    manifest["manifest_sha256_without_self"] = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
