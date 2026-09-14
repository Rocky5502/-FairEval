import json
from pathlib import Path

import pytest

from faireval.cli import main
from faireval.datasets.base import DatasetAdapter, DatasetCard
from faireval.freeze import freeze_dataset, verify_freeze
from faireval.schema import Item, UserInstance


class _ToyAdapter(DatasetAdapter):
    dataset_id = "toy"

    def __init__(self) -> None:
        self._manifest = {"dataset_id": "toy"}

    def card(self) -> DatasetCard:
        return DatasetCard(
            dataset_id="toy",
            source="unit-test",
            version="1",
            license="test-only",
            domain="toy",
            track="test",
            observed_fields=("interactions",),
        )

    def build_instances(
        self,
        raw_dir: Path,
        *,
        users: int,
        candidate_set_size: int,
        max_history_items: int,
        seed: int,
    ):
        del raw_dir, max_history_items
        candidates = [Item(f"i{i}", f"Item {i}") for i in range(candidate_set_size)]
        emitted = min(users, 2)
        for user_index in range(emitted):
            yield UserInstance(
                dataset="toy",
                user_id=f"u{user_index}",
                history=[Item("h1", "History", {"rating": 5})],
                candidates=candidates,
                relevant_item_ids=frozenset({"i0"}),
                demographics={"group": "A"},
                instance_metadata={"seed": seed},
            )
        self._manifest = {"dataset_id": "toy", "emitted_users": emitted, "seed": seed}

    def preprocessing_manifest(self):
        return dict(self._manifest)


def test_freeze_is_deterministic_and_verifiable(tmp_path: Path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = freeze_dataset(
        _ToyAdapter(),
        tmp_path,
        first_dir,
        users=2,
        candidate_set_size=4,
        max_history_items=3,
        seed=1729,
    )
    second = freeze_dataset(
        _ToyAdapter(),
        tmp_path,
        second_dir,
        users=2,
        candidate_set_size=4,
        max_history_items=3,
        seed=1729,
    )

    assert first["instances_sha256"] == second["instances_sha256"]
    assert first["instance_count"] == 2
    verified = verify_freeze(first_dir)
    assert verified["verification"] == "PASS"
    assert verified["verified_instance_count"] == 2


def test_verify_freeze_detects_tampering(tmp_path: Path):
    output = tmp_path / "freeze"
    freeze_dataset(
        _ToyAdapter(),
        tmp_path,
        output,
        users=1,
        candidate_set_size=3,
        max_history_items=2,
        seed=7,
    )
    with (output / "instances.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema_version": "faireval-instance-v1", "tampered": True}) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_freeze(output)


def test_cli_prints_dataset_card(capsys):
    rc = main(["dataset-card", "--dataset", "movielens_1m"])
    assert rc == 0
    card = json.loads(capsys.readouterr().out)
    assert card["dataset_id"] == "movielens_1m"
    assert "gender" in card["observed_fields"]
