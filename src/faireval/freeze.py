from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .datasets.base import DatasetAdapter
from .schema import Item, PersonalityProfile, UserInstance


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    # Numpy scalar support without importing numpy into the serialization layer.
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    raise TypeError(f"cannot serialize {type(value).__name__}: {value!r}")


def _item_dict(item: Item) -> dict[str, Any]:
    return {
        "item_id": str(item.item_id),
        "title": str(item.title),
        "metadata": _json_safe(item.metadata),
    }


def _personality_dict(profile: PersonalityProfile | None) -> dict[str, float] | None:
    return None if profile is None else profile.as_dict()


def instance_to_dict(instance: UserInstance) -> dict[str, Any]:
    instance.validate()
    return {
        "schema_version": "faireval-instance-v1",
        "dataset": instance.dataset,
        "user_id": str(instance.user_id),
        "history": [_item_dict(item) for item in instance.history],
        "candidates": [_item_dict(item) for item in instance.candidates],
        "relevant_item_ids": sorted(str(value) for value in instance.relevant_item_ids),
        "demographics": _json_safe(instance.demographics),
        "personality": _personality_dict(instance.personality),
        "instance_metadata": _json_safe(instance.instance_metadata),
    }


def canonical_json(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def freeze_dataset(
    adapter: DatasetAdapter,
    raw_dir: Path,
    output_dir: Path,
    *,
    users: int,
    candidate_set_size: int,
    max_history_items: int,
    seed: int,
) -> dict[str, Any]:
    """Materialize one deterministic benchmark split and its manifest.

    This function intentionally exhausts the adapter before writing the manifest
    so preprocessing counts/hashes describe the exact materialized instances.
    The instance file and manifest are written atomically to avoid a partial
    freeze being mistaken for a valid benchmark artifact.
    """
    instances = list(
        adapter.build_instances(
            raw_dir,
            users=users,
            candidate_set_size=candidate_set_size,
            max_history_items=max_history_items,
            seed=seed,
        )
    )
    if not instances:
        raise ValueError(f"adapter {adapter.dataset_id!r} emitted no instances")

    instance_rows = [instance_to_dict(instance) for instance in instances]
    instance_path = output_dir / "instances.jsonl"
    _atomic_write_lines(instance_path, (canonical_json(row) for row in instance_rows))
    instance_hash = file_sha256(instance_path)

    card = asdict(adapter.card())
    manifest: dict[str, Any] = {
        "schema_version": "faireval-freeze-v1",
        "dataset_id": adapter.dataset_id,
        "dataset_card": _json_safe(card),
        "build_parameters": {
            "users_requested": int(users),
            "candidate_set_size": int(candidate_set_size),
            "max_history_items": int(max_history_items),
            "seed": int(seed),
        },
        "instance_count": len(instance_rows),
        "instances_filename": instance_path.name,
        "instances_sha256": instance_hash,
        "preprocessing_manifest": _json_safe(adapter.preprocessing_manifest()),
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write_lines(manifest_path, [json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)])
    # Hash after writing; keep it outside the manifest to avoid a self-hash cycle.
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def verify_freeze(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    instance_path = output_dir / "instances.jsonl"
    if not manifest_path.is_file() or not instance_path.is_file():
        raise FileNotFoundError("freeze requires manifest.json and instances.jsonl")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("instances_sha256")
    actual_hash = file_sha256(instance_path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"instance hash mismatch: expected {expected_hash}, actual {actual_hash}"
        )

    count = 0
    with instance_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "faireval-instance-v1":
                raise ValueError(f"instances.jsonl:{line_no}: unexpected schema version")
            count += 1
    if count != int(manifest.get("instance_count", -1)):
        raise ValueError(
            f"instance count mismatch: manifest={manifest.get('instance_count')}, actual={count}"
        )
    return {
        **manifest,
        "verification": "PASS",
        "verified_instances_sha256": actual_hash,
        "verified_instance_count": count,
    }
