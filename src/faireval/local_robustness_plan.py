from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .freeze import canonical_json
from .local_plan import load_local_model_panel
from .robustness_plan import compile_rq3_extra_plan


LOCAL_RQ3_FAMILIES = {"qwen25_local", "phi35_local"}


def _seed_for_cell(row: Mapping[str, Any], *, experiment_seed: int) -> int:
    """Derive a deterministic local generation seed from the complete RQ3 cell.

    Robustness cells can differ only in cue, cutoff, template, candidate order,
    temperature, or repetition. Hashing the complete semantic cell prevents two
    distinct factor levels from accidentally sharing the same local RNG seed.
    """
    payload = dict(row)
    payload.pop("cell_id", None)
    payload.pop("seed", None)
    material = {
        "experiment_seed": int(experiment_seed),
        "semantic_cell": payload,
    }
    digest = hashlib.sha256(canonical_json(material).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def freeze_local_robustness_seeds(
    cells: Sequence[Mapping[str, Any]], *, experiment_seed: int
) -> list[dict[str, Any]]:
    """Freeze a generation seed into every local robustness cell and rehash it."""
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in cells:
        row = dict(raw)
        row.pop("cell_id", None)
        family = str(row.get("model_family", ""))
        if family not in LOCAL_RQ3_FAMILIES:
            raise ValueError(f"unexpected local robustness model family {family!r}")
        if row.get("output_token_parameter") != "max_new_tokens":
            raise ValueError("local robustness cells must freeze max_new_tokens semantics")
        row["seed"] = _seed_for_cell(row, experiment_seed=experiment_seed)
        cell_id = hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()
        if cell_id in seen:
            raise ValueError("local robustness seed freeze produced duplicate cell IDs")
        seen.add(cell_id)
        row["cell_id"] = cell_id
        output.append(row)
    return output


def compile_local_rq3_extra_plan(
    *,
    freeze_root: Path,
    counterfactuals_yaml: Path,
    local_models_yaml: Path,
    study_design_yaml: Path,
    experiment_yaml: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compile the registered RQ3 robustness plan for the two local models.

    The semantic factor geometry is delegated to the same planner used by the
    hosted panel. Only the model panel and seeded local-generation contract are
    different. This preserves direct comparability while keeping hosted and
    local inference strata separate in analysis and reporting.
    """
    base_cells, base_manifest = compile_rq3_extra_plan(
        freeze_root=freeze_root,
        counterfactuals_yaml=counterfactuals_yaml,
        models_yaml=local_models_yaml,
        study_design_yaml=study_design_yaml,
        experiment_yaml=experiment_yaml,
        seed=seed,
        model_panel_loader=load_local_model_panel,
    )
    cells = freeze_local_robustness_seeds(base_cells, experiment_seed=seed)
    families = {str(row["model_family"]) for row in cells}
    if families != LOCAL_RQ3_FAMILIES:
        raise AssertionError(f"local RQ3 plan has unexpected families: {sorted(families)}")
    if any(row.get("seed") is None for row in cells):
        raise AssertionError("every local RQ3 cell must have a frozen generation seed")

    manifest = {
        **base_manifest,
        "schema_version": "faireval-local-rq3-plan-manifest-v1",
        "reporting_scope": "separate_local_open_weight_whitebox_stratum",
        "local_generation_seeds_frozen": True,
        "model_families": sorted(families),
        "shared_registered_geometry_with_hosted_rq3": True,
        "hosted_style_semantic_plan_sha256_before_local_seed_freeze": base_manifest.get(
            "plan_sha256"
        ),
        "plan_sha256": hashlib.sha256(canonical_json(cells).encode("utf-8")).hexdigest(),
    }
    return cells, manifest
