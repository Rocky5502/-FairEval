from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dataset_acquisition.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError("dataset acquisition config must contain a datasets mapping")
    return payload


def inspect_dataset(dataset_id: str, raw_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    required_all = [str(name) for name in spec.get("required_all", [])]
    required_any = [str(name) for name in spec.get("required_any_of", [])]
    missing_all = [name for name in required_all if not (raw_dir / name).is_file()]
    present_any = [name for name in required_any if (raw_dir / name).is_file()]

    fallback_ok = False
    if required_any and not present_any and bool(spec.get("allow_single_spreadsheet_fallback", False)):
        candidates = []
        if raw_dir.is_dir():
            candidates = [
                path.name
                for path in raw_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".csv"}
            ]
        fallback_ok = len(candidates) == 1
        if fallback_ok:
            present_any = candidates

    ready = raw_dir.is_dir() and not missing_all and (not required_any or bool(present_any))
    return {
        "dataset": dataset_id,
        "raw_dir": str(raw_dir),
        "directory_exists": raw_dir.is_dir(),
        "required_all": required_all,
        "missing_required_all": missing_all,
        "required_any_of": required_any,
        "present_any_of": present_any,
        "single_spreadsheet_fallback_used": fallback_ok,
        "ready_for_release_hash": ready,
        "source": spec.get("source"),
        "acquisition": spec.get("acquisition"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate FairEval raw dataset directories against adapter file contracts"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", action="append", help="dataset ID; repeat to check a subset")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero unless every selected dataset satisfies its raw-file contract",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    specs = config["datasets"]
    selected = list(args.dataset or specs.keys())
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise ValueError(f"unknown dataset IDs: {unknown}")

    rows = []
    for dataset_id in selected:
        spec = specs[dataset_id]
        if not isinstance(spec, dict):
            raise ValueError(f"dataset acquisition spec for {dataset_id} must be a mapping")
        raw_dir = Path(args.data_root) / "raw" / dataset_id
        rows.append(inspect_dataset(dataset_id, raw_dir, spec))

    report = {
        "schema_version": "faireval-raw-layout-audit-v1",
        "selected_datasets": selected,
        "ready_count": sum(bool(row["ready_for_release_hash"]) for row in rows),
        "total_count": len(rows),
        "all_ready_for_release_hash": all(bool(row["ready_for_release_hash"]) for row in rows),
        "datasets": rows,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if args.strict and not report["all_ready_for_release_hash"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
