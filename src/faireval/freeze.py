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


def _item_from_dict(row: Mapping[str, Any], *, field: str) -> Item:
    if "item_id" not in row or "title" not in row:
        raise ValueError(f"{field} item must contain item_id and title")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{field}.metadata must be an object")
    return Item(
        item_id=str(row["item_id"]),
        title=str(row["title"]),
        metadata=dict(metadata),
    )


def instance_from_dict(row: Mapping[str, Any]) -> UserInstance:
    """Reconstruct and validate one frozen benchmark instance."""
    if row.get("schema_version") != "faireval-instance-v1":
        raise ValueError(f"unexpected instance schema version: {row.get('schema_version')!r}")
    for required in ("dataset", "user_id", "history", "candidates", "relevant_item_ids"):
        if required not in row:
            raise ValueError(f"frozen instance missing required field {required!r}")

    history_raw = row["history"]
    candidates_raw = row["candidates"]
    relevant_raw = row["relevant_item_ids"]
    if not isinstance(history_raw, list) or not isinstance(candidates_raw, list):
        raise ValueError("history and candidates must be arrays")
    if not isinstance(relevant_raw, list):
        raise ValueError("relevant_item_ids must be an array")

    personality_raw = row.get("personality")
    personality = None
    if personality_raw is not None:
        if not isinstance(personality_raw, Mapping):
            raise ValueError("personality must be an object or null")
        expected = {
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        }
        if set(personality_raw) != expected:
            raise ValueError(
                "personality must contain exactly the five OCEAN fields; "
                f"found={sorted(personality_raw)!r}"
            )
        personality = PersonalityProfile(
            openness=float(personality_raw["openness"]),
            conscientiousness=float(personality_raw["conscientiousness"]),
            extraversion=float(personality_raw["extraversion"]),
            agreeableness=float(personality_raw["agreeableness"]),
            neuroticism=float(personality_raw["neuroticism"]),
        )
        personality.as_dict()

    demographics = row.get("demographics", {})
    metadata = row.get("instance_metadata", {})
    if not isinstance(demographics, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("demographics and instance_metadata must be objects")

    instance = UserInstance(
        dataset=str(row["dataset"]),
        user_id=str(row["user_id"]),
        history=[_item_from_dict(item, field="history") for item in history_raw],
        candidates=[_item_from_dict(item, field="candidates") for item in candidates_raw],
        relevant_item_ids=frozenset(str(value) for value in relevant_raw),
        demographics=dict(demographics),
        personality=personality,
        instance_metadata=dict(metadata),
    )
    instance.validate()
    return instance


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
    _atomic_write_lines(
        manifest_path,
        [json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)],
    )
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
            # Validate structure while verifying so a hash-consistent but malformed
            # artifact cannot enter an experiment plan.
            instance_from_dict(row)
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


def load_frozen_instances(output_dir: Path) -> list[UserInstance]:
    """Load instances only after verifying their manifest hash and structure."""
    verify_freeze(output_dir)
    instance_path = output_dir / "instances.jsonl"
    instances: list[UserInstance] = []
    with instance_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                instances.append(instance_from_dict(json.loads(line)))
    return instances
