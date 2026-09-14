import hashlib
import json
from pathlib import Path

from faireval.conditions import preference_only
from faireval.datasets.base import DatasetAdapter, DatasetCard
from faireval.execute import execute_plan, load_and_verify_plan
from faireval.freeze import canonical_json, file_sha256, freeze_dataset
from faireval.plan import PlannedCondition, expand_core_run_cells
from faireval.providers.base import GenerationRequest, GenerationResponse, ProviderAdapter
from faireval.schema import Item, UserInstance


class _OneUserAdapter(DatasetAdapter):
    dataset_id = "movielens_1m"

    def card(self):
        return DatasetCard(
            dataset_id=self.dataset_id,
            source="test",
            version="1",
            license="test",
            domain="movies",
            track="test",
            observed_fields=("ratings",),
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
        del raw_dir, users, candidate_set_size, max_history_items, seed
        yield UserInstance(
            dataset=self.dataset_id,
            user_id="u1",
            history=[Item("h", "History")],
            candidates=[Item("a", "A"), Item("b", "B"), Item("c", "C")],
            relevant_item_ids=frozenset({"a"}),
            demographics={"gender": "female", "age_group": "25_34"},
        )


class _FakeProvider(ProviderAdapter):
    family = "fake"
    provider_name = "fake_provider"

    def __init__(self):
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        return GenerationResponse(
            text=json.dumps({"ranked_item_ids": ["a", "b"]}),
            requested_model_id=request.model_id,
            resolved_model_version="fake-v1",
            provider_metadata={"test": True},
        )

    def supports_seed(self) -> bool:
        return False


def _write_one_cell_plan(plan_dir: Path) -> dict:
    planned = PlannedCondition(
        dataset="movielens_1m",
        user_id="u1",
        condition=preference_only(),
        analysis_roles=("test",),
        confirmatory=True,
    )
    cells = expand_core_run_cells(
        [planned],
        model_panel=[{"family": "fake", "model_id": "fake-model"}],
        repetitions=1,
        k=2,
    )
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / "run_plan.jsonl"
    with plan_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in cells:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "faireval-core-plan-manifest-v1",
        "planned_api_cells": len(cells),
        "plan_sha256": hashlib.sha256(
            "\n".join(canonical_json(row) for row in cells).encode("utf-8")
        ).hexdigest(),
        "run_plan_file_sha256": file_sha256(plan_path),
    }
    (plan_dir / "plan_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return cells[0]


def test_execute_plan_runs_once_then_resumes_without_duplicate(tmp_path: Path):
    freeze_root = tmp_path / "freeze"
    freeze_dataset(
        _OneUserAdapter(),
        tmp_path,
        freeze_root / "movielens_1m",
        users=1,
        candidate_set_size=3,
        max_history_items=1,
        seed=1,
    )
    plan_dir = tmp_path / "plan"
    planned = _write_one_cell_plan(plan_dir)
    output = tmp_path / "results.jsonl"
    provider = _FakeProvider()

    first = execute_plan(
        plan_dir=plan_dir,
        freeze_root=freeze_root,
        output_jsonl=output,
        code_commit_sha="abc123",
        provider_builder=lambda family: provider,
    )
    assert first["executed_cells"] == 1
    assert first["remaining_plan_cells"] == 0
    assert provider.calls == 1

    second = execute_plan(
        plan_dir=plan_dir,
        freeze_root=freeze_root,
        output_jsonl=output,
        code_commit_sha="abc123",
        provider_builder=lambda family: provider,
    )
    assert second["executed_cells"] == 0
    assert provider.calls == 1

    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["planned_cell_id"] == planned["cell_id"]
    assert row["code_commit_sha"] == "abc123"
    assert row["final_valid"] is True


def test_executor_rejects_tampered_plan(tmp_path: Path):
    plan_dir = tmp_path / "plan"
    _write_one_cell_plan(plan_dir)
    with (plan_dir / "run_plan.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    try:
        load_and_verify_plan(plan_dir)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tampered plan should not verify")
