from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .freeze import canonical_json, file_sha256, load_frozen_instances
from .plan import (
    DEMOGRAPHIC_DATASETS,
    GENERALIZATION_ONLY_DATASETS,
    PERSONALITY_DATASETS,
    PlannedCondition,
    load_model_panel,
    plan_core_conditions,
)
from .schema import PromptCondition, UserInstance


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stable_subset(
    instances: Sequence[UserInstance], *, n: int, seed: int, label: str
) -> set[str]:
    scored = []
    for instance in instances:
        user_id = str(instance.user_id)
        token = f"{seed}|rq3|{label}|{instance.dataset}|{user_id}".encode("utf-8")
        scored.append((hashlib.sha256(token).hexdigest(), user_id))
    return {user_id for _, user_id in sorted(scored)[: min(n, len(scored))]}


def _condition_dict(condition: PromptCondition) -> dict[str, Any]:
    return {
        "condition_id": condition.condition_id,
        "condition_name": condition.condition_name,
        "demographics": condition.demographics,
        "personality": None if condition.personality is None else condition.personality.as_dict(),
        "intervention": dict(condition.intervention),
    }


def _representative(planned: PlannedCondition) -> bool:
    condition_id = planned.condition.condition_id
    if planned.dataset in PERSONALITY_DATASETS:
        return condition_id in {"C3", "C4"}
    if planned.dataset in DEMOGRAPHIC_DATASETS:
        return condition_id == "C1" or (
            condition_id.startswith("C2:") and planned.confirmatory
        )
    if planned.dataset in GENERALIZATION_ONLY_DATASETS:
        return condition_id == "C0"
    return False


def _cell(
    planned: PlannedCondition,
    model: Mapping[str, str],
    *,
    factor: str,
    level: str,
    template_id: str,
    cue_id: str,
    candidate_order_seed: int | None,
    k: int,
    repetition: int,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": "faireval-run-plan-v1",
        "dataset": planned.dataset,
        "user_id": planned.user_id,
        "condition": _condition_dict(planned.condition),
        "analysis_roles": ["rq3_reliability", f"rq3_{factor}"],
        "confirmatory": False,
        "model_family": str(model["family"]),
        "model_id": str(model["model_id"]),
        "reasoning_or_thinking_setting": str(model["reasoning_or_thinking_setting"]),
        "sampling_policy": str(model["sampling_policy"]),
        "output_token_parameter": str(model["output_token_parameter"]),
        "template_id": template_id,
        "prompt_mode": "audit",
        "cue_id": cue_id,
        "candidate_order_seed": candidate_order_seed,
        "k": int(k),
        "repetition": int(repetition),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
        "robustness_factor": factor,
        "robustness_level": level,
    }
    row["cell_id"] = _digest(row)
    return row


def compile_rq3_extra_plan(
    *,
    freeze_root: Path,
    counterfactuals_yaml: Path,
    models_yaml: Path,
    study_design_yaml: Path,
    experiment_yaml: Path,
    seed: int,
    model_panel_loader: Callable[[Path], list[dict[str, str]]] = load_model_panel,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compile only RQ3 cells not already present in the primary core plan.

    The primary core cell supplies template A, structured cue, adapter-frozen
    candidate order, K=10 and the main temperature. This extra plan schedules
    only missing factor levels under a one-factor-at-a-time design.

    ``model_panel_loader`` defaults to the six-family hosted panel. The explicit
    hook lets the separately reported local white-box stratum reuse exactly the
    same registered robustness geometry without duplicating the design code.
    Local callers remain responsible for freezing generation seeds and rehashing
    cell IDs after this shared semantic plan is constructed.
    """
    counterfactual_cfg = yaml.safe_load(counterfactuals_yaml.read_text(encoding="utf-8"))
    study = yaml.safe_load(study_design_yaml.read_text(encoding="utf-8"))
    experiment = yaml.safe_load(experiment_yaml.read_text(encoding="utf-8"))
    if not isinstance(counterfactual_cfg, Mapping):
        raise ValueError("counterfactual config must be a mapping")
    if not isinstance(study, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError("study/experiment configs must be mappings")

    model_panel = model_panel_loader(models_yaml)
    primary = study["primary_prompt"]
    generation = study["primary_generation"]
    robustness = experiment["robustness"]

    primary_template = str(primary["template_id"])
    primary_cue = str(primary["cue_id"])
    primary_k = int(primary["k"])
    main_temperature = float(generation["temperature_requested"])
    main_top_p = float(generation["top_p_requested"])
    max_output_tokens = int(generation["max_output_tokens"])

    cells: list[dict[str, Any]] = []
    dataset_counts: dict[str, dict[str, int]] = {}
    all_datasets = sorted(PERSONALITY_DATASETS | DEMOGRAPHIC_DATASETS | GENERALIZATION_ONLY_DATASETS)

    for dataset in all_datasets:
        instances = load_frozen_instances(freeze_root / dataset)
        conditions = [
            row
            for row in plan_core_conditions(
                instances,
                counterfactual_config=counterfactual_cfg,
                seed=seed,
                one_trait_subset_users=0,
                demographic_robustness_subset_users=0,
            )
            if _representative(row)
        ]

        factor_counts: dict[str, int] = {}

        def selected_conditions(factor: str, n: int) -> list[PlannedCondition]:
            selected_ids = _stable_subset(instances, n=n, seed=seed, label=factor)
            return [row for row in conditions if row.user_id in selected_ids]

        # Prompt wording: template A is reused from core; schedule B/C only.
        prompt_cfg = robustness["task_paraphrase"]
        prompt_rows = selected_conditions("prompt", int(prompt_cfg["users_per_dataset"]))
        extra_templates = [x for x in prompt_cfg["templates"] if str(x) != primary_template]
        before = len(cells)
        for planned in prompt_rows:
            for model in model_panel:
                for template_id in extra_templates:
                    cells.append(
                        _cell(
                            planned,
                            model,
                            factor="prompt",
                            level=str(template_id),
                            template_id=str(template_id),
                            cue_id=primary_cue,
                            candidate_order_seed=None,
                            k=primary_k,
                            repetition=0,
                            temperature=main_temperature,
                            top_p=main_top_p,
                            max_output_tokens=max_output_tokens,
                        )
                    )
        factor_counts["prompt"] = len(cells) - before

        # Cue realization applies only to demographic C1/C2 cells.
        if dataset in set(robustness["demographic_cue"]["datasets"]):
            cue_cfg = robustness["demographic_cue"]
            cue_rows = selected_conditions("cue", int(cue_cfg["users_per_dataset"]))
            cue_rows = [
                row
                for row in cue_rows
                if row.condition.condition_id == "C1" or row.condition.condition_id.startswith("C2:")
            ]
            extra_cues = [x for x in cue_cfg["cues"] if str(x) != primary_cue]
            before = len(cells)
            for planned in cue_rows:
                for model in model_panel:
                    for cue_id in extra_cues:
                        cells.append(
                            _cell(
                                planned,
                                model,
                                factor="cue",
                                level=str(cue_id),
                                template_id=primary_template,
                                cue_id=str(cue_id),
                                candidate_order_seed=None,
                                k=primary_k,
                                repetition=0,
                                temperature=main_temperature,
                                top_p=main_top_p,
                                max_output_tokens=max_output_tokens,
                            )
                        )
            factor_counts["cue"] = len(cells) - before

        order_cfg = robustness["candidate_order"]
        order_rows = selected_conditions("candidate_order", int(order_cfg["users_per_dataset"]))
        before = len(cells)
        for planned in order_rows:
            for model in model_panel:
                for order_seed in order_cfg["order_seeds"]:
                    cells.append(
                        _cell(
                            planned,
                            model,
                            factor="candidate_order",
                            level=f"seed_{int(order_seed)}",
                            template_id=primary_template,
                            cue_id=primary_cue,
                            candidate_order_seed=int(order_seed),
                            k=primary_k,
                            repetition=0,
                            temperature=main_temperature,
                            top_p=main_top_p,
                            max_output_tokens=max_output_tokens,
                        )
                    )
        factor_counts["candidate_order"] = len(cells) - before

        cutoff_cfg = robustness["cutoff"]
        cutoff_rows = selected_conditions("cutoff", int(cutoff_cfg["users_per_dataset"]))
        extra_k = [int(value) for value in cutoff_cfg["k_values"] if int(value) != primary_k]
        before = len(cells)
        for planned in cutoff_rows:
            for model in model_panel:
                for k in extra_k:
                    cells.append(
                        _cell(
                            planned,
                            model,
                            factor="cutoff",
                            level=f"k_{k}",
                            template_id=primary_template,
                            cue_id=primary_cue,
                            candidate_order_seed=None,
                            k=k,
                            repetition=0,
                            temperature=main_temperature,
                            top_p=main_top_p,
                            max_output_tokens=max_output_tokens,
                        )
                    )
        factor_counts["cutoff"] = len(cells) - before

        stochastic_cfg = robustness["stochasticity"]
        stochastic_rows = selected_conditions(
            "stochasticity", int(stochastic_cfg["users_per_dataset"])
        )
        before = len(cells)
        for planned in stochastic_rows:
            for model in model_panel:
                for repetition in range(int(stochastic_cfg["repetitions"])):
                    cells.append(
                        _cell(
                            planned,
                            model,
                            factor="stochasticity",
                            level="high_stochasticity",
                            template_id=primary_template,
                            cue_id=primary_cue,
                            candidate_order_seed=None,
                            k=primary_k,
                            repetition=repetition,
                            temperature=float(stochastic_cfg["temperature_requested_where_supported"]),
                            top_p=main_top_p,
                            max_output_tokens=max_output_tokens,
                        )
                    )
        factor_counts["stochasticity"] = len(cells) - before
        dataset_counts[dataset] = factor_counts

    ids = [str(row["cell_id"]) for row in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("RQ3 extra plan generated duplicate cell IDs")

    manifest = {
        "schema_version": "faireval-rq3-plan-manifest-v1",
        "seed": int(seed),
        "baseline_reuse": {
            "template": primary_template,
            "cue": primary_cue,
            "candidate_order": "adapter_frozen_order",
            "k": primary_k,
            "main_generation_repetition": 0,
        },
        "planned_api_cells": len(cells),
        "dataset_factor_cell_counts": dataset_counts,
        "plan_sha256": _digest(cells),
        "config_sha256": {
            "counterfactuals": file_sha256(counterfactuals_yaml),
            "models": file_sha256(models_yaml),
            "study_design": file_sha256(study_design_yaml),
            "experiment": file_sha256(experiment_yaml),
        },
    }
    return cells, manifest


def write_rq3_plan(
    output_dir: Path, cells: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "run_plan.jsonl"
    plan_path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in cells),
        encoding="utf-8",
    )
    payload = dict(manifest)
    payload["run_plan_file_sha256"] = file_sha256(plan_path)
    manifest_path = output_dir / "plan_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"plan": plan_path, "manifest": manifest_path}
