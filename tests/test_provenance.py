from pathlib import Path

import pytest

from faireval.provenance import canonical_directory_manifest


def test_raw_release_hash_is_order_independent_and_content_sensitive(tmp_path: Path):
    root = tmp_path / "release"
    (root / "nested").mkdir(parents=True)
    (root / "b.txt").write_text("beta\n", encoding="utf-8")
    (root / "nested" / "a.txt").write_text("alpha\n", encoding="utf-8")

    first = canonical_directory_manifest(root)
    second = canonical_directory_manifest(root)
    assert first["directory_sha256"] == second["directory_sha256"]
    assert [row["path"] for row in first["files"]] == ["b.txt", "nested/a.txt"]
    assert first["file_count"] == 2

    (root / "b.txt").write_text("changed\n", encoding="utf-8")
    changed = canonical_directory_manifest(root)
    assert changed["directory_sha256"] != first["directory_sha256"]


def test_raw_release_hash_rejects_empty_directory(tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(ValueError, match="empty"):
        canonical_directory_manifest(root)


def test_raw_release_hash_rejects_symlinks_when_supported(tmp_path: Path):
    root = tmp_path / "release"
    root.mkdir()
    target = root / "target.txt"
    target.write_text("x", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    with pytest.raises(ValueError, match="symlink"):
        canonical_directory_manifest(root)
