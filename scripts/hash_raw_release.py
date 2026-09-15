from __future__ import annotations

import argparse
import json
from pathlib import Path

from faireval.provenance import write_directory_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hash an exact FairEval raw dataset release into a canonical manifest."
    )
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    manifest = write_directory_manifest(
        Path(args.raw_dir),
        Path(args.output_manifest),
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "directory_sha256": manifest["directory_sha256"],
                "manifest_path": str(Path(args.output_manifest)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
