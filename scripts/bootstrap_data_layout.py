from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ROOT / "configs" / "datasets.yaml"
DEFAULT_RELEASES = ROOT / "configs" / "dataset_releases.yaml"


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create the FairEval six-dataset raw/processed directory layout without "
            "silently downloading or redistributing third-party datasets."
        )
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets-config", default=str(DEFAULT_DATASETS))
    parser.add_argument("--release-config", default=str(DEFAULT_RELEASES))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    datasets_cfg = _load(Path(args.datasets_config))
    release_cfg = _load(Path(args.release_config))
    datasets = datasets_cfg.get("datasets", {})
    releases = release_cfg.get("datasets", {})
    if not isinstance(datasets, dict) or not isinstance(releases, dict):
        raise ValueError("both YAML files must contain a datasets mapping")

    rows: list[dict[str, Any]] = []
    for dataset_id, spec in datasets.items():
        if not isinstance(spec, dict):
            raise ValueError(f"dataset spec for {dataset_id} must be a mapping")
        raw_dir = data_root / "raw" / dataset_id
        processed_dir = data_root / "processed" / dataset_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        lock = releases.get(dataset_id, {})
        if not isinstance(lock, dict):
            lock = {}
        row = {
            "dataset": dataset_id,
            "domain": spec.get("domain"),
            "track": spec.get("track"),
            "source": spec.get("source"),
            "license_declared_in_dataset_config": spec.get("license"),
            "release_status": lock.get("status", "missing_release_lock"),
            "release_id": lock.get("release_id"),
            "license_reviewed": bool(lock.get("license_reviewed", False)),
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "next_step": (
                f"Obtain the exact upstream release from {spec.get('source')}, review its terms, "
                f"then run: python scripts/hash_raw_release.py --dataset {dataset_id} "
                f"--raw-dir {raw_dir}"
            ),
        }
        rows.append(row)

    report = {
        "schema_version": "faireval-data-bootstrap-v1",
        "data_root": str(data_root),
        "datasets": rows,
        "note": (
            "Raw third-party data are intentionally gitignored. This command creates the "
            "canonical local layout and provenance instructions; it does not bypass licenses, "
            "authentication, click-through terms, or redistribution restrictions."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
