from __future__ import annotations

import pytest

from scripts.hosted_paired_extension_balanced450 import (
    EXPECTED_CALLS,
    FAMILIES,
    TARGET_USERS,
    USERS_PER_IDENTITY_GROUP,
    _assert_geometry,
)


def _manifest() -> dict:
    return {
        "users_per_identity_group": 6,
        "target_users_total": 18,
        "target_cells_total_in_parent_plan": 450,
        "target_cells_completed_before_extension": 0,
        "planned_api_cells": 450,
        "target_identity_group_counts": {"A": 6, "B": 6, "C": 6},
        "scientific_outcomes_inspected_before_extension_definition": False,
        "run_schema_version": "faireval-run-v7",
        "prompt_interface_version": "faireval-prompt-interface-v7",
        "model_families": sorted(FAMILIES),
        "projected_incremental_cost_with_safety_rmb": 162.72,
    }


def test_balanced450_geometry_is_frozen() -> None:
    assert USERS_PER_IDENTITY_GROUP == 6
    assert TARGET_USERS == 18
    assert EXPECTED_CALLS == 450
    assert set(FAMILIES) == {"meta", "qwen", "deepseek", "openai", "anthropic"}
    _assert_geometry(_manifest())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("users_per_identity_group", 5),
        ("target_users_total", 17),
        ("target_cells_total_in_parent_plan", 449),
        ("target_cells_completed_before_extension", 1),
        ("planned_api_cells", 449),
    ),
)
def test_balanced450_rejects_geometry_drift(field: str, value: int) -> None:
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(RuntimeError):
        _assert_geometry(manifest)


def test_balanced450_rejects_identity_imbalance() -> None:
    manifest = _manifest()
    manifest["target_identity_group_counts"] = {"A": 7, "B": 6, "C": 5}
    with pytest.raises(RuntimeError, match="identity balance"):
        _assert_geometry(manifest)


def test_balanced450_rejects_family_drift() -> None:
    manifest = _manifest()
    manifest["model_families"] = ["meta", "qwen", "deepseek", "openai", "google"]
    with pytest.raises(RuntimeError, match="families drift"):
        _assert_geometry(manifest)


def test_balanced450_rejects_cost_projection_above_target() -> None:
    manifest = _manifest()
    manifest["projected_incremental_cost_with_safety_rmb"] = 170.01
    with pytest.raises(RuntimeError, match="exceeds frozen 170 RMB"):
        _assert_geometry(manifest)
