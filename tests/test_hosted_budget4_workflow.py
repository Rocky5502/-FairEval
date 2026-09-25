from __future__ import annotations

import pytest

from scripts.hosted_paired_extension_budget4 import EXPECTED_CALLS, FAMILIES, _assert_geometry


def _manifest() -> dict:
    return {
        "target_users_total": 9,
        "target_cells_total_in_parent_plan": 180,
        "target_cells_completed_before_extension": 0,
        "planned_api_cells": 180,
        "target_identity_group_counts": {"A": 3, "B": 3, "C": 3},
        "scientific_outcomes_inspected_before_extension_definition": False,
        "run_schema_version": "faireval-run-v7",
        "prompt_interface_version": "faireval-prompt-interface-v7",
        "model_families": sorted(FAMILIES),
        "projected_incremental_cost_with_safety_rmb": 35.89,
    }


def test_budget4_geometry_is_frozen() -> None:
    assert EXPECTED_CALLS == 180
    assert set(FAMILIES) == {"meta", "qwen", "deepseek", "openai"}
    _assert_geometry(_manifest())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_users_total", 8),
        ("target_cells_total_in_parent_plan", 179),
        ("target_cells_completed_before_extension", 1),
        ("planned_api_cells", 179),
    ),
)
def test_budget4_rejects_geometry_drift(field: str, value: int) -> None:
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(RuntimeError):
        _assert_geometry(manifest)


def test_budget4_rejects_family_drift() -> None:
    manifest = _manifest()
    manifest["model_families"] = ["meta", "qwen", "deepseek", "anthropic"]
    with pytest.raises(RuntimeError, match="families drift"):
        _assert_geometry(manifest)


def test_budget4_rejects_cost_projection_above_target() -> None:
    manifest = _manifest()
    manifest["projected_incremental_cost_with_safety_rmb"] = 40.01
    with pytest.raises(RuntimeError, match="exceeds 40 RMB"):
        _assert_geometry(manifest)
