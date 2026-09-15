from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .conditions import (
    all_categorical_demographic_counterfactuals,
    build_one_trait_donor_map,
    build_personality_derangement,
    observed_demographic,
    one_trait_counterfactual,
    preference_only,
    shuffled_personality,
    true_personality,
)
from .freeze import canonical_json, file_sha256, load_frozen_instances
from .schema import OCEAN_KEYS, PromptCondition, UserInstance


PERSONALITY_DATASETS = {"personality2018", "music_master_bfi2", "reasoner"}
DEMOGRAPHIC_DATASETS = {"movielens_1m", "lastfm_1k"}
GENERALIZATION_ONLY_DATASETS = {"mind"}


@dataclass(frozen=True)
class PlannedCondition:
    dataset: str
    user_id: str
    condition: PromptCondition
    analysis_roles: tuple[str, ...]
    confirmatory: bool


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stable_subset(
    instances: Sequence[UserInstance],
    *,
    n: int,
    seed: int,
    label: str,
) -> set[str]:
    if n < 0:
        raise ValueError("subset size must be non-negative")
    scored = []
    for instance in instances:
        user_id = str(instance.user_id)
        digest = hashlib.sha256(f"{seed}|{label}|{instance.dataset}|{user_id}".encode("utf-8")).hexdigest()
        scored.append((digest, user_id))
    return {user_id for _, user_id in sorted(scored)[: min(n, len(scored))]}


def _condition_dict(condition: PromptCondition) -> dict[str, Any]:
    personality = None
    if condition.personality is not None:
        personality = condition.personality.as_dict()
    return {
        "condition_id": condition.condition_id,
        "condition_name": condition.condition_name,
        "demographics": condition.demographics,
        "personality": personality,
        "intervention": dict(condition.intervention),
    }


def _deduplicate_conditions(rows: Sequence[PlannedCondition]) -> list[PlannedCondition]:
    seen: set[str] = set()
    output: list[PlannedCondition] = []
    for row in rows:
        key = _sha256_json(
            {
                "dataset": row.dataset,
                "user_id": row.user_id,
                "condition": _condition_dict(row.condition),
                "analysis_roles": row.analysis_roles,
                "confirmatory": row.confirmatory,
            }
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def plan_core_conditions(
    instances: Sequence[UserInstance],
    *,
    counterfactual_config: Mapping[str, Any],
    seed: int,
    one_trait_subset_users: int = 30,
    demographic_robustness_subset_users: int = 20,
) -> list[PlannedCondition]:
    if not instances:
        return []
    datasets = {instance.dataset for instance in instances}
    if len(datasets) != 1:
        raise ValueError("plan_core_conditions expects one dataset at a time")
    dataset = next(iter(datasets))
    ids = [str(instance.user_id) for instance in instances]
    if len(set(ids)) != len(ids):
        raise ValueError("frozen instances must have unique user IDs")

    rows: list[PlannedCondition] = []
    if dataset in GENERALIZATION_ONLY_DATASETS:
        for instance in instances:
            rows.append(
                PlannedCondition(
                    dataset=dataset,
                    user_id=str(instance.user_id),
                    condition=preference_only(),
                    analysis_roles=("rq3_generalization",),
                    confirmatory=False,
                )
            )
        return rows

    if dataset in PERSONALITY_DATASETS:
        donor_map = build_personality_derangement(instances, seed=seed)
        by_id = {str(instance.user_id): instance for instance in instances}
        one_trait_users = _stable_subset(
            instances,
            n=one_trait_subset_users,
            seed=seed,
            label="one_trait_subset",
        )
        trait_maps = {
            trait: build_one_trait_donor_map(instances, trait=trait, seed=seed)
            for trait in OCEAN_KEYS
        }
        for instance in instances:
            user_id = str(instance.user_id)
            rows.append(
                PlannedCondition(
                    dataset=dataset,
                    user_id=user_id,
                    condition=preference_only(),
                    analysis_roles=("rq2_personality",),
                    confirmatory=True,
                )
            )
            rows.append(
                PlannedCondition(
                    dataset=dataset,
                    user_id=user_id,
                    condition=true_personality(instance),
                    analysis_roles=("rq2_personality",),
                    confirmatory=True,
                )
            )
            rows.append(
                PlannedCondition(
                    dataset=dataset,
                    user_id=user_id,
                    condition=shuffled_personality(
                        instance,
                        donor_instance=by_id[donor_map[user_id]],
                    ),
                    analysis_roles=("rq2_personality",),
                    confirmatory=True,
                )
            )
            if user_id in one_trait_users:
                for trait in OCEAN_KEYS:
                    donor_id = trait_maps[trait][user_id]
                    rows.append(
                        PlannedCondition(
                            dataset=dataset,
                            user_id=user_id,
                            condition=one_trait_counterfactual(
                                instance,
                                trait=trait,
                                donor_instance=by_id[donor_id],
                            ),
                            analysis_roles=("rq2_trait_ablation",),
                            confirmatory=False,
                        )
                    )

    elif dataset in DEMOGRAPHIC_DATASETS:
        dataset_cfg = counterfactual_config.get(dataset)
        if not isinstance(dataset_cfg, Mapping):
            raise ValueError(f"counterfactual config missing dataset {dataset!r}")
        confirmatory_cfg = dataset_cfg.get("confirmatory", {})
        robustness_cfg = dataset_cfg.get("robustness", {})
        if not isinstance(confirmatory_cfg, Mapping) or not isinstance(robustness_cfg, Mapping):
            raise ValueError("counterfactual config sections must be mappings")

        robustness_users = _stable_subset(
            instances,
            n=demographic_robustness_subset_users,
            seed=seed,
            label="demographic_robustness_subset",
        )

        for instance in instances:
            user_id = str(instance.user_id)
            rows.append(
                PlannedCondition(
                    dataset=dataset,
                    user_id=user_id,
                    condition=preference_only(),
                    analysis_roles=("rq1_demographic", "rq4_mitigation_baseline"),
                    confirmatory=True,
                )
            )
            rows.append(
                PlannedCondition(
                    dataset=dataset,
                    user_id=user_id,
                    condition=observed_demographic(instance),
                    analysis_roles=("rq1_demographic", "rq4_mitigation_baseline"),
                    confirmatory=True,
                )
            )
            for attribute, spec in confirmatory_cfg.items():
                if not isinstance(spec, Mapping) or "values" not in spec:
                    raise ValueError(f"invalid confirmatory spec for {dataset}.{attribute}")
                for condition in all_categorical_demographic_counterfactuals(
                    instance,
                    attribute=str(attribute),
                    allowed_values=tuple(spec["values"]),
                ):
                    rows.append(
                        PlannedCondition(
                            dataset=dataset,
                            user_id=user_id,
                            condition=condition,
                            analysis_roles=("rq1_demographic", "rq4_mitigation_baseline"),
                            confirmatory=True,
                        )
                    )

            if user_id in robustness_users:
                for attribute, spec in robustness_cfg.items():
                    if not isinstance(spec, Mapping) or "values" not in spec:
                        raise ValueError(f"invalid robustness spec for {dataset}.{attribute}")
                    for condition in all_categorical_demographic_counterfactuals(
                        instance,
                        attribute=str(attribute),
                        allowed_values=tuple(spec["values"]),
                    ):
                        rows.append(
                            PlannedCondition(
                                dataset=dataset,
                                user_id=user_id,
                                condition=condition,
                                analysis_roles=("rq1_demographic_robustness",),
                                confirmatory=False,
                            )
                        )

    elif dataset not in GENERALIZATION_ONLY_DATASETS:
        raise ValueError(f"dataset {dataset!r} is not assigned to a FairEval study track")

    return _deduplicate_conditions(rows)


def load_model_panel(models_yaml: Path) -> list[dict[str, str]]:
    """Load the six-family model panel including executable provider semantics."""
    config = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    models = config.get("models") if isinstance(config, Mapping) else None
    if not isinstance(models, list):
        raise ValueError("models YAML must contain a models list")
    output: list[dict[str, str]] = []
    families: set[str] = set()
    for row in models:
        if not isinstance(row, Mapping) or not row.get("enabled", True):
            continue
        family = str(row.get("family", "")).strip()
        model_id = str(row.get("model_id", "")).strip()
        reasoning = str(row.get("reasoning_or_thinking_setting", "")).strip()
        sampling_policy = str(row.get("sampling_policy", "")).strip()
        output_token_parameter = str(row.get("output_token_parameter", "")).strip()
        if not family or not model_id:
            raise ValueError("each enabled model needs family and model_id")
        if not reasoning:
            raise ValueError(f"enabled model {family!r} needs reasoning_or_thinking_setting")
        if not sampling_policy:
            raise ValueError(f"enabled model {family!r} needs sampling_policy")
        if not output_token_parameter:
            raise ValueError(f"enabled model {family!r} needs output_token_parameter")
        if family in families:
            raise ValueError(f"duplicate enabled family {family!r}")
        families.add(family)
        output.append(
            {
                "family": family,
                "model_id": model_id,
                "reasoning_or_thinking_setting": reasoning,
                "sampling_policy": sampling_policy,
                "output_token_parameter": output_token_parameter,
            }
        )
    if len(output) != 6:
        raise ValueError(f"FairEval core panel requires exactly six enabled families, found {len(output)}")
    return output


def expand_core_run_cells(
    conditions: Sequence[PlannedCondition],
    *,
    model_panel: Sequence[Mapping[str, str]],
    repetitions: int = 3,
    k: int = 10,
    template_id: str = "field_v2_a",
    cue_id: str = "structured_key_value",
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_output_tokens: int = 512,
) -> list[dict[str, Any]]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    rows: list[dict[str, Any]] = []
    for planned in conditions:
        for model in model_panel:
            for repetition in range(repetitions):
                row: dict[str, Any] = {
                    "schema_version": "faireval-run-plan-v1",
                    "dataset": planned.dataset,
                    "user_id": planned.user_id,
                    "condition": _condition_dict(planned.condition),
                    "analysis_roles": list(planned.analysis_roles),
                    "confirmatory": planned.confirmatory,
                    "model_family": str(model["family"]),
                    "model_id": str(model["model_id"]),
                    "reasoning_or_thinking_setting": str(model["reasoning_or_thinking_setting"]),
                    "sampling_policy": str(model["sampling_policy"]),
                    "output_token_parameter": str(model["output_token_parameter"]),
                    "template_id": template_id,
                    "prompt_mode": "audit",
                    "cue_id": cue_id,
                    "candidate_order_seed": None,
                    "k": int(k),
                    "repetition": int(repetition),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "max_output_tokens": int(max_output_tokens),
                }
                row["cell_id"] = _sha256_json(row)
                rows.append(row)
    return rows


def compile_core_plan(
    *,
    freeze_root: Path,
    counterfactuals_yaml: Path,
    models_yaml: Path,
    seed: int,
    one_trait_subset_users: int = 30,
    demographic_robustness_subset_users: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counterfactual_cfg = yaml.safe_load(counterfactuals_yaml.read_text(encoding="utf-8"))
    if not isinstance(counterfactual_cfg, Mapping):
        raise ValueError("counterfactual configuration must be a mapping")
    model_panel = load_model_panel(models_yaml)

    all_conditions: list[PlannedCondition] = []
    dataset_manifests: dict[str, Any] = {}
    for dataset in sorted(PERSONALITY_DATASETS | DEMOGRAPHIC_DATASETS | GENERALIZATION_ONLY_DATASETS):
        dataset_dir = freeze_root / dataset
        instances = load_frozen_instances(dataset_dir)
        conditions = plan_core_conditions(
            instances,
            counterfactual_config=counterfactual_cfg,
            seed=seed,
            one_trait_subset_users=one_trait_subset_users,
            demographic_robustness_subset_users=demographic_robustness_subset_users,
        )
        all_conditions.extend(conditions)
        manifest_path = dataset_dir / "freeze_manifest.json"
        dataset_manifests[dataset] = {
            "freeze_manifest_sha256": file_sha256(manifest_path),
            "n_frozen_users": len(instances),
            "n_planned_conditions": len(conditions),
        }

    cells = expand_core_run_cells(all_conditions, model_panel=model_panel)
    manifest = {
        "schema_version": "faireval-run-plan-manifest-v1",
        "seed": seed,
        "planned_conditions": len(all_conditions),
        "planned_api_cells": len(cells),
        "datasets": dataset_manifests,
        "models_yaml_sha256": file_sha256(models_yaml),
        "counterfactuals_yaml_sha256": file_sha256(counterfactuals_yaml),
        "plan_sha256": _sha256_json(cells),
    }
    return cells, manifest


def write_core_plan(
    *,
    output_dir: Path,
    cells: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "run_plan.jsonl"
    manifest_path = output_dir / "plan_manifest.json"

    plan_text = "".join(canonical_json(dict(row)) + "\n" for row in cells)
    plan_path.write_text(plan_text, encoding="utf-8")
    payload = dict(manifest)
    payload["run_plan_file_sha256"] = file_sha256(plan_path)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"plan": plan_path, "manifest": manifest_path}
