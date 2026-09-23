from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .conditions import (
    all_categorical_demographic_counterfactuals,
    build_personality_derangement,
    observed_demographic,
    preference_only,
    shuffled_personality,
    true_personality,
)
from .freeze import canonical_json, file_sha256, load_frozen_instances
from .plan import (
    DEMOGRAPHIC_DATASETS,
    GENERALIZATION_ONLY_DATASETS,
    PERSONALITY_DATASETS,
    PlannedCondition,
    expand_core_run_cells,
    plan_core_conditions,
)
from .schema import UserInstance


FAIRSYNTH_IDENTITY_VALUES = ("A", "B", "C")


def load_local_model_panel(path: Path) -> list[dict[str, str]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = config.get("models") if isinstance(config, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("local_models.yaml must contain a models list")
    output: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("enabled", True):
            continue
        payload = {
            "family": str(row.get("family", "")).strip(),
            "model_id": str(row.get("model_id", "")).strip(),
            "reasoning_or_thinking_setting": str(
                row.get("reasoning_or_thinking_setting", "")
            ).strip(),
            "sampling_policy": str(row.get("sampling_policy", "")).strip(),
            "output_token_parameter": str(row.get("output_token_parameter", "")).strip(),
            "quantization_policy": str(row.get("quantization_policy", "none")).strip(),
            "attn_implementation": str(row.get("attn_implementation", "")).strip(),
            "kv_cache_policy": str(row.get("kv_cache_policy", "")).strip(),
        }
        required = ("family", "model_id", "reasoning_or_thinking_setting", "sampling_policy", "output_token_parameter", "quantization_policy", "attn_implementation", "kv_cache_policy")
        if not all(payload[key] for key in required):
            raise ValueError(f"incomplete local model row: {row!r}")
        if payload["output_token_parameter"] != "max_new_tokens":
            raise ValueError("direct Transformers local models must use max_new_tokens")
        output.append(payload)
    if {row["family"] for row in output} != {"qwen25_local", "phi35_local"}:
        raise ValueError("local track requires exactly qwen25_local and phi35_local")
    return output


def _stable_subset(
    instances: Sequence[UserInstance],
    *,
    n: int,
    seed: int,
    label: str,
    offset: int = 0,
) -> list[UserInstance]:
    if n < 0 or offset < 0:
        raise ValueError("subset size and offset must be non-negative")
    scored = []
    for instance in instances:
        digest = hashlib.sha256(
            f"{seed}|{label}|{instance.dataset}|{instance.user_id}".encode("utf-8")
        ).hexdigest()
        scored.append((digest, instance))
    ordered = [instance for _, instance in sorted(scored, key=lambda row: row[0])]
    return ordered[offset : offset + min(n, max(0, len(ordered) - offset))]


def _cell_seed(cell: Mapping[str, Any], *, experiment_seed: int) -> int:
    raw = "|".join(
        [
            str(experiment_seed),
            str(cell["dataset"]),
            str(cell["user_id"]),
            str(cell["condition"]["condition_id"]),
            str(cell["model_family"]),
            str(cell["repetition"]),
            str(cell["template_id"]),
            str(cell["candidate_order_seed"]),
        ]
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big", signed=False)


def _freeze_local_seeds(cells: Sequence[Mapping[str, Any]], *, experiment_seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in cells:
        row = dict(source)
        row.pop("cell_id", None)
        row["seed"] = _cell_seed(row, experiment_seed=experiment_seed)
        row["cell_id"] = hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()
        output.append(row)
    return output


def plan_fairsynth_conditions(
    instances: Sequence[UserInstance],
    *,
    seed: int,
) -> list[PlannedCondition]:
    if not instances:
        return []
    if {instance.dataset for instance in instances} != {"fairsynth360"}:
        raise ValueError("plan_fairsynth_conditions only accepts fairsynth360 instances")

    donor_map = build_personality_derangement(instances, seed=seed)
    by_id = {str(instance.user_id): instance for instance in instances}
    rows: list[PlannedCondition] = []
    for instance in instances:
        user_id = str(instance.user_id)
        rows.append(
            PlannedCondition(
                dataset="fairsynth360",
                user_id=user_id,
                condition=preference_only(),
                analysis_roles=("synthetic_identity_sanity", "synthetic_personality_sanity"),
                confirmatory=False,
            )
        )
        rows.append(
            PlannedCondition(
                dataset="fairsynth360",
                user_id=user_id,
                condition=observed_demographic(instance),
                analysis_roles=("synthetic_identity_sanity",),
                confirmatory=False,
            )
        )
        for condition in all_categorical_demographic_counterfactuals(
            instance,
            attribute="synthetic_identity_group",
            allowed_values=FAIRSYNTH_IDENTITY_VALUES,
        ):
            rows.append(
                PlannedCondition(
                    dataset="fairsynth360",
                    user_id=user_id,
                    condition=condition,
                    analysis_roles=("synthetic_identity_sanity",),
                    confirmatory=False,
                )
            )
        rows.append(
            PlannedCondition(
                dataset="fairsynth360",
                user_id=user_id,
                condition=true_personality(instance),
                analysis_roles=("synthetic_personality_sanity",),
                confirmatory=False,
            )
        )
        rows.append(
            PlannedCondition(
                dataset="fairsynth360",
                user_id=user_id,
                condition=shuffled_personality(
                    instance,
                    donor_instance=by_id[donor_map[user_id]],
                ),
                analysis_roles=("synthetic_personality_sanity",),
                confirmatory=False,
            )
        )
    return rows


def compile_local_open_weight_plan(
    *,
    freeze_root: Path,
    counterfactuals_yaml: Path,
    local_models_yaml: Path,
    seed: int,
    include_real_world: bool = True,
    fairsynth_users: int = 360,
    fairsynth_user_offset: int = 0,
    repetitions: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    local_models = load_local_model_panel(local_models_yaml)
    counterfactual_cfg = yaml.safe_load(counterfactuals_yaml.read_text(encoding="utf-8"))
    if not isinstance(counterfactual_cfg, Mapping):
        raise ValueError("counterfactual configuration must be a mapping")

    conditions: list[PlannedCondition] = []
    datasets: dict[str, Any] = {}
    if include_real_world:
        for dataset in sorted(PERSONALITY_DATASETS | DEMOGRAPHIC_DATASETS | GENERALIZATION_ONLY_DATASETS):
            dataset_dir = freeze_root / dataset
            instances = load_frozen_instances(dataset_dir)
            planned = plan_core_conditions(
                instances,
                counterfactual_config=counterfactual_cfg,
                seed=seed,
            )
            conditions.extend(planned)
            datasets[dataset] = {
                "manifest_sha256": file_sha256(dataset_dir / "manifest.json"),
                "instances": len(instances),
                "conditions": len(planned),
                "scope": "real_world_replication",
            }

    synth_dir = freeze_root / "fairsynth360"
    synth_all = load_frozen_instances(synth_dir)
    synth_selected = _stable_subset(
        synth_all,
        n=fairsynth_users,
        seed=seed,
        label="local_fairsynth_subset",
        offset=fairsynth_user_offset,
    )
    synth_conditions = plan_fairsynth_conditions(synth_selected, seed=seed)
    conditions.extend(synth_conditions)
    datasets["fairsynth360"] = {
        "manifest_sha256": file_sha256(synth_dir / "manifest.json"),
        "instances_available": len(synth_all),
        "instances_selected": len(synth_selected),
        "selection_offset": int(fairsynth_user_offset),
        "conditions": len(synth_conditions),
        "scope": "synthetic_controlled_stress_test",
    }

    cells = expand_core_run_cells(
        conditions,
        model_panel=local_models,
        repetitions=repetitions,
        k=10,
        temperature=0.2,
        top_p=1.0,
        max_output_tokens=512,
    )
    cells = _freeze_local_seeds(cells, experiment_seed=seed)
    manifest = {
        "schema_version": "faireval-local-open-weight-plan-v1",
        "seed": int(seed),
        "local_generation_seeds_frozen": True,
        "planned_conditions": len(conditions),
        "planned_api_cells": len(cells),
        "model_families": [row["family"] for row in local_models],
        "datasets": datasets,
        "local_models_yaml_sha256": file_sha256(local_models_yaml),
        "counterfactuals_yaml_sha256": file_sha256(counterfactuals_yaml),
        "reporting_scope": "separate_local_open_weight_track",
    }
    return cells, manifest
