from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .freeze import canonical_json, file_sha256, load_frozen_instances
from .providers.base import ProviderAdapter
from .providers.factory import build_provider
from .runner import run_one
from .schema import PersonalityProfile, PromptCondition, UserInstance


def _cell_digest(row_without_cell_id: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(row_without_cell_id).encode("utf-8")).hexdigest()


def _condition_from_dict(row: Mapping[str, Any]) -> PromptCondition:
    for required in ("condition_id", "condition_name", "intervention"):
        if required not in row:
            raise ValueError(f"planned condition missing {required!r}")
    demographics_raw = row.get("demographics")
    if demographics_raw is not None and not isinstance(demographics_raw, Mapping):
        raise ValueError("planned demographics must be an object or null")

    personality_raw = row.get("personality")
    personality = None
    if personality_raw is not None:
        if not isinstance(personality_raw, Mapping):
            raise ValueError("planned personality must be an object or null")
        personality = PersonalityProfile(
            openness=float(personality_raw["openness"]),
            conscientiousness=float(personality_raw["conscientiousness"]),
            extraversion=float(personality_raw["extraversion"]),
            agreeableness=float(personality_raw["agreeableness"]),
            neuroticism=float(personality_raw["neuroticism"]),
        )
        personality.as_dict()

    intervention = row["intervention"]
    if not isinstance(intervention, Mapping):
        raise ValueError("planned intervention must be an object")
    return PromptCondition(
        condition_id=str(row["condition_id"]),
        condition_name=str(row["condition_name"]),
        demographics=None if demographics_raw is None else dict(demographics_raw),
        personality=personality,
        intervention=dict(intervention),
    )


def load_and_verify_plan(plan_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a run plan only if its file hash and every cell ID are intact."""
    plan_path = plan_dir / "run_plan.jsonl"
    manifest_path = plan_dir / "plan_manifest.json"
    if not plan_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("plan directory requires run_plan.jsonl and plan_manifest.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_file_hash = manifest.get("run_plan_file_sha256")
    actual_file_hash = file_sha256(plan_path)
    if expected_file_hash != actual_file_hash:
        raise ValueError(
            f"run-plan file hash mismatch: expected {expected_file_hash}, actual {actual_file_hash}"
        )

    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    with plan_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"run_plan.jsonl:{line_no}: row must be an object")
            if row.get("schema_version") != "faireval-run-plan-v1":
                raise ValueError(f"run_plan.jsonl:{line_no}: unsupported schema version")
            cell_id = str(row.get("cell_id", ""))
            if not cell_id:
                raise ValueError(f"run_plan.jsonl:{line_no}: missing cell_id")
            payload = dict(row)
            payload.pop("cell_id", None)
            expected_cell_id = _cell_digest(payload)
            if cell_id != expected_cell_id:
                raise ValueError(
                    f"run_plan.jsonl:{line_no}: cell_id mismatch; "
                    f"expected {expected_cell_id}, got {cell_id}"
                )
            if cell_id in seen:
                raise ValueError(f"run_plan.jsonl:{line_no}: duplicate cell_id {cell_id}")
            seen.add(cell_id)
            cells.append(row)

    if len(cells) != int(manifest.get("planned_api_cells", -1)):
        raise ValueError(
            "planned cell-count mismatch: "
            f"manifest={manifest.get('planned_api_cells')}, file={len(cells)}"
        )
    return cells, manifest


def completed_cell_ids(output_jsonl: Path) -> set[str]:
    """Return successfully persisted plan IDs and reject duplicate output rows."""
    if not output_jsonl.exists():
        return set()
    completed: set[str] = set()
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            cell_id = row.get("planned_cell_id")
            if not cell_id:
                continue
            cell_id = str(cell_id)
            if cell_id in completed:
                raise ValueError(
                    f"{output_jsonl}:{line_no}: duplicate completed planned_cell_id {cell_id}"
                )
            completed.add(cell_id)
    return completed


def _load_instance_index(freeze_root: Path, datasets: Sequence[str]) -> dict[tuple[str, str], UserInstance]:
    index: dict[tuple[str, str], UserInstance] = {}
    for dataset in sorted(set(datasets)):
        for instance in load_frozen_instances(freeze_root / dataset):
            key = (dataset, str(instance.user_id))
            if key in index:
                raise ValueError(f"duplicate frozen instance key {key!r}")
            index[key] = instance
    return index


def pending_cells(
    cells: Sequence[Mapping[str, Any]],
    *,
    completed: set[str],
    families: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select pending cells deterministically without changing the frozen plan."""
    output: list[dict[str, Any]] = []
    for raw in cells:
        row = dict(raw)
        if str(row["cell_id"]) in completed:
            continue
        if families is not None and str(row["model_family"]) not in families:
            continue
        output.append(row)
    return output


def _verify_provider_contract(row: Mapping[str, Any], provider: ProviderAdapter) -> None:
    """Reject runtime provider semantics that disagree with the immutable plan."""
    cell_id = str(row.get("cell_id", "<unknown>"))
    expected_token_field = str(row.get("output_token_parameter", "")).strip()
    if not expected_token_field:
        raise ValueError(f"cell {cell_id} is missing frozen output_token_parameter")
    actual_token_field = str(getattr(provider, "output_token_parameter", "")).strip()
    if not actual_token_field:
        raise ValueError(
            f"provider {provider.family!r} does not expose output_token_parameter; "
            "cannot prove runtime/plan API compatibility"
        )
    if actual_token_field != expected_token_field:
        raise ValueError(
            f"cell {cell_id} output-token field drift: "
            f"plan={expected_token_field!r}, runtime={actual_token_field!r}"
        )

    optional_contracts = {
        "quantization_policy": str(getattr(provider, "quantization_policy", "")).strip(),
        "attn_implementation": str(getattr(provider, "attn_implementation", "") or "").strip(),
        "kv_cache_policy": (
            "disabled_for_transformers_compatibility"
            if bool(getattr(provider, "disable_kv_cache", False))
            else "enabled"
        ),
    }
    for key, actual in optional_contracts.items():
        if key not in row:
            continue
        expected = str(row.get(key, "")).strip()
        if expected != actual:
            raise ValueError(
                f"cell {cell_id} {key} drift: plan={expected!r}, runtime={actual!r}"
            )


def execute_plan(
    *,
    plan_dir: Path,
    freeze_root: Path,
    output_jsonl: Path,
    code_commit_sha: str,
    families: set[str] | None = None,
    max_cells: int | None = None,
    provider_builder: Callable[[str], ProviderAdapter] = build_provider,
    before_cell: Callable[[Mapping[str, Any], int], None] | None = None,
    after_cell: Callable[[Mapping[str, Any], int], None] | None = None,
) -> dict[str, Any]:
    """Execute pending frozen cells sequentially with exact resume semantics.

    This function does not modify the run plan. A provider is instantiated only
    if at least one pending cell needs that family. A cell counts as completed
    after ``run_one`` has persisted its row, even if the model output is invalid;
    invalidity is an experimental outcome and should not be silently regenerated
    until it becomes valid. Core hosted plans omit ``seed``; local open-weight
    plans may freeze one and the executor forwards it exactly.

    ``before_cell``/``after_cell`` are intentionally narrow hooks used by the
    hosted budget guard. They preserve the immutable plan while allowing a hard
    pre-request spend check and post-request ledger refresh.
    """
    if not code_commit_sha.strip():
        raise ValueError("code_commit_sha is required for planned execution")
    if max_cells is not None and max_cells <= 0:
        raise ValueError("max_cells must be positive when supplied")

    cells, manifest = load_and_verify_plan(plan_dir)
    completed = completed_cell_ids(output_jsonl)
    known_ids = {str(row["cell_id"]) for row in cells}
    unknown_completed = completed - known_ids
    if unknown_completed:
        raise ValueError(
            "output contains planned cell IDs that are not in this plan; "
            "use a separate output file for a different plan"
        )

    selected = pending_cells(cells, completed=completed, families=families)
    if max_cells is not None:
        selected = selected[:max_cells]

    instance_index = _load_instance_index(
        freeze_root,
        [str(row["dataset"]) for row in selected],
    )
    providers: dict[str, ProviderAdapter] = {}
    executed = 0

    for row in selected:
        if before_cell is not None:
            before_cell(row, executed)

        dataset = str(row["dataset"])
        user_id = str(row["user_id"])
        key = (dataset, user_id)
        if key not in instance_index:
            raise ValueError(f"planned cell references missing frozen instance {key!r}")

        family = str(row["model_family"])
        provider = providers.get(family)
        if provider is None:
            provider = provider_builder(family)
            providers[family] = provider
        _verify_provider_contract(row, provider)

        condition_raw = row["condition"]
        if not isinstance(condition_raw, Mapping):
            raise ValueError(f"cell {row['cell_id']} condition must be an object")
        condition = _condition_from_dict(condition_raw)

        reasoning_setting = str(row.get("reasoning_or_thinking_setting", "")).strip()
        if not reasoning_setting:
            raise ValueError(
                f"cell {row['cell_id']} is missing frozen reasoning_or_thinking_setting"
            )
        planned_seed = None if row.get("seed") is None else int(row["seed"])

        run_one(
            instance=instance_index[key],
            condition=condition,
            provider=provider,
            model_id=str(row["model_id"]),
            k=int(row["k"]),
            repetition=int(row["repetition"]),
            output_jsonl=output_jsonl,
            temperature=float(row["temperature"]),
            top_p=float(row["top_p"]),
            max_output_tokens=int(row["max_output_tokens"]),
            template_id=str(row["template_id"]),
            prompt_mode=str(row["prompt_mode"]),
            cue_id=str(row["cue_id"]),
            candidate_order_seed=(
                None
                if row.get("candidate_order_seed") is None
                else int(row["candidate_order_seed"])
            ),
            seed=planned_seed,
            reasoning_or_thinking_setting=reasoning_setting,
            code_commit_sha=code_commit_sha,
            planned_cell_id=str(row["cell_id"]),
            run_schema_version=str(row.get("run_schema_version", "faireval-run-v5")),
            prompt_interface_version=str(
                row.get("prompt_interface_version", "faireval-prompt-interface-v5")
            ),
        )
        executed += 1

        if after_cell is not None:
            after_cell(row, executed)

    completed_after = completed_cell_ids(output_jsonl)
    return {
        "schema_version": "faireval-execution-summary-v1",
        "plan_sha256": manifest.get("plan_sha256"),
        "run_plan_file_sha256": manifest.get("run_plan_file_sha256"),
        "selected_pending_cells": len(selected),
        "executed_cells": executed,
        "completed_plan_cells_total": len(completed_after),
        "planned_api_cells_total": len(cells),
        "remaining_plan_cells": len(cells) - len(completed_after),
        "family_filter": None if families is None else sorted(families),
        "code_commit_sha": code_commit_sha,
    }
