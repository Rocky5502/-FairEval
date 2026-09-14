from __future__ import annotations

import hashlib
import json
from collections import defaultdict
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
from .freeze import canonical_json, load_frozen_instances
from .schema import OCEAN_KEYS, PromptCondition, UserInstance


PERSONALITY_DATASETS = {"personality2018", "music_master_bfi2", "reasoner"}
DEMOGRAPHIC_DATASETS = {"movielens_1m", "lastfm_1k"}
GENERALIZATION_ONLY_DATASETS = {"mind"}


def _stable_int(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _sha256_json(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def _condition_dict(condition: PromptCondition) -> dict[str, Any]:
    return {
        "condition_id": condition.condition_id,
        "condition_name": condition.condition_name,
        "demographics": None if condition.demographics is None else dict(condition.demographics),
        "personality": None if condition.personality is None else condition.personality.as_dict(),
        "intervention": dict(condition.intervention),
    }


def _stable_subset(
    instances: Sequence[UserInstance],
    *,
    n: int,
    seed: int,
    label: str,
) -> set[str]:
    if n <= 0:
        return set()
    ordered = sorted(
        instances,
        key=lambda instance: (
            _stable_int(seed, label, instance.dataset, instance.user_id),
            str(instance.user_id),
        ),
    )
    return {str(instance.user_id) for instance in ordered[: min(n, len(ordered))]}


@dataclass(frozen=True)
class PlannedCondition:
    dataset: str
    user_id: str
    condition: PromptCondition
    analysis_roles: tuple[str, ...]
    confirmatory: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "user_id": self.user_id,
            "condition": _condition_dict(self.condition),
            "analysis_roles": list(self.analysis_roles),
            "confirmatory": self.confirmatory,
        }


def _deduplicate_conditions(rows: Sequence[PlannedCondition]) -> list[PlannedCondition]:
    """Merge identical run conditions while retaining all analysis roles."""
    merged: dict[tuple[str, str, str], PlannedCondition] = {}
    role_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    confirmatory: dict[tuple[str, str, str], bool] = defaultdict(bool)
    for row in rows:
        key = (row.dataset, row.user_id, canonical_json(_condition_dict(row.condition)))
        role_sets[key].update(row.analysis_roles)
        confirmatory[key] = confirmatory[key] or row.confirmatory
        merged.setdefault(key, row)
    output = []
    for key, original in merged.items():
        output.append(
            PlannedCondition(
                dataset=original.dataset,
                user_id=original.user_id,
                condition=original.condition,
                analysis_roles=tuple(sorted(role_sets[key])),
                confirmatory=confirmatory[key],
            )
        )
    return sorted(
        output,
        key=lambda row: (row.dataset, row.user_id, row.condition.condition_id),
    )


def plan_core_conditions(
    instances: Sequence[UserInstance],
    *,
    counterfactual_config: Mapping[str, Any],
    seed: int,
    one_trait_subset_users: int = 30,
    demographic_robustness_subset_users: int = 20,
) -> list[PlannedCondition]:
    """Compile condition-level cells for the confirmatory/core experiment.

    The function refuses cross-dataset cohorts. RQ1 and RQ2 are deliberately
    factorized: demographic conditions are not generated on personality-only
    datasets, and personality claims are not generated from demographic-only or
    MIND instances.
    """
    if not instances:
        raise ValueError("instances cannot be empty")
    datasets = {instance.dataset for instance in instances}
    if len(datasets) != 1:
        raise ValueError(f"plan one dataset at a time, got {sorted(datasets)!r}")
    dataset = next(iter(datasets))
    by_id = {str(instance.user_id): instance for instance in instances}
    if len(by_id) != len(instances):
        raise ValueError("user IDs must be unique within a frozen dataset")

    rows: list[PlannedCondition] = []

    # Preference-only is useful across every dataset and can be reused by more
    # than one RQ without paying for duplicate API calls.
    for instance in instances:
        roles = ["rq3_generalization"]
        if dataset in PERSONALITY_DATASETS:
            roles.append("rq2_personality")
        if dataset in DEMOGRAPHIC_DATASETS:
            roles.extend(["rq1_demographic", "rq4_mitigation_baseline"])
        rows.append(
            PlannedCondition(
                dataset=dataset,
                user_id=str(instance.user_id),
                condition=preference_only(),
                analysis_roles=tuple(roles),
                confirmatory=dataset != GENERALIZATION_ONLY_DATASETS,
            )
        )

    if dataset in PERSONALITY_DATASETS:
        donor_map = build_personality_derangement(instances, seed=seed)
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
        if not family or not model_id:
            raise ValueError("each enabled model needs family and model_id")
        if family in families:
            raise ValueError(f"duplicate enabled family {family!r}")
        families.add(family)
        output.append({"family": family, "model_id": model_id})
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
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"missing frozen dataset directory {dataset_dir}")
        instances = load_frozen_instances(dataset_dir)
        all_conditions.extend(
            plan_core_conditions(
                instances,
                counterfactual_config=counterfactual_cfg,
                seed=seed,
                one_trait_subset_users=one_trait_subset_users,
                demographic_robustness_subset_users=demographic_robustness_subset_users,
            )
        )
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        dataset_manifests[dataset] = {
            "instances_sha256": manifest["instances_sha256"],
            "instance_count": manifest["instance_count"],
        }

    cells = expand_core_run_cells(all_conditions, model_panel=model_panel)
    ids = [row["cell_id"] for row in cells]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate run-plan cell IDs detected")

    manifest = {
        "schema_version": "faireval-core-plan-manifest-v1",
        "seed": int(seed),
        "datasets": dataset_manifests,
        "model_panel": model_panel,
        "planned_condition_rows": len(all_conditions),
        "planned_api_cells": len(cells),
        "plan_sha256": hashlib.sha256(
            "\n".join(canonical_json(row) for row in cells).encode("utf-8")
        ).hexdigest(),
    }
    return cells, manifest
