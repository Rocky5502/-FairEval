from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from faireval.datasets.fairsynth360 import FEATURE_NAMES, synthetic_items, synthetic_users


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the deterministic project-owned FairSynth-360 dataset to auditable CSV files"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/fairsynth360-v1"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    users_path = out / "users.csv"
    with users_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "user_id",
                "synthetic_identity_group",
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
                *[f"latent_pref_{name}" for name in FEATURE_NAMES],
            ]
        )
        for user in synthetic_users():
            p = user.personality
            writer.writerow(
                [
                    user.user_id,
                    user.identity_group,
                    f"{p.openness:.8f}",
                    f"{p.conscientiousness:.8f}",
                    f"{p.extraversion:.8f}",
                    f"{p.agreeableness:.8f}",
                    f"{p.neuroticism:.8f}",
                    *[f"{value:.8f}" for value in user.preference_vector],
                ]
            )

    items_path = out / "items.csv"
    with items_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "title", *FEATURE_NAMES])
        for item in synthetic_items():
            writer.writerow(
                [item.item_id, item.title, *[f"{value:.8f}" for value in item.features]]
            )

    manifest = {
        "schema_version": "faireval-fairsynth-export-v1",
        "dataset": "FairSynth-360",
        "version": "v1",
        "users": 360,
        "items": 120,
        "project_generated": True,
        "external_personal_data": False,
        "identity_semantics": "balanced_semantically_meaningless_A_B_C",
        "identity_independent_of_relevance_by_construction": True,
        "synthetic_ocean_is_human_measurement": False,
        "files": {
            "users.csv": {"sha256": _sha256(users_path), "bytes": users_path.stat().st_size},
            "items.csv": {"sha256": _sha256(items_path), "bytes": items_path.stat().st_size},
        },
        "license": "release license must be frozen before public dataset publication",
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
