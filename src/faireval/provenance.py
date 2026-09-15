from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_directory_manifest(root: Path) -> dict[str, Any]:
    """Return a deterministic manifest for an exact raw dataset directory.

    Every regular file is included. Symlinks are rejected because their target
    resolution is environment-dependent and could make two nominally identical
    releases hash differently. Relative POSIX paths, byte sizes and file SHA-256
    digests are committed into the directory digest.
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw release directory does not exist: {root}")

    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"raw release contains symlink; materialize it first: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )

    if not files:
        raise ValueError(f"raw release directory is empty: {root}")

    canonical_files = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    directory_sha256 = hashlib.sha256(canonical_files).hexdigest()
    return {
        "schema_version": "faireval-raw-release-manifest-v1",
        "hash_definition": "sha256(canonical_json(sorted(relative_path,size_bytes,file_sha256)))",
        "root_name": root.name,
        "file_count": len(files),
        "total_bytes": sum(int(row["size_bytes"]) for row in files),
        "directory_sha256": directory_sha256,
        "files": files,
    }


def write_directory_manifest(root: Path, output_json: Path) -> dict[str, Any]:
    manifest = canonical_directory_manifest(root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
