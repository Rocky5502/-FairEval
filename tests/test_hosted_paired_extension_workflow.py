from __future__ import annotations

import pytest

from scripts.hosted_paired_extension import _assert_geometry


def _manifest() -> dict:
    return {
        "target_users_total": 9,
        "target_cells_total_in_parent_plan": 324,
        "target_cells_completed_before_extension": 72,
        "planned_api_cells": 252,
        "target_identity_group_counts": {"A": 3, "B": 3, "C": 3},
        "scientific_outcomes_inspected_before_extension_definition": False,
        "projected_incremental_cost_with_safety_rmb": 186.98,
    }


def test_hosted_paired_workflow_accepts_frozen_geometry() -> None:
    _assert_geometry(_manifest())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_users_total", 10),
        ("target_cells_total_in_parent_plan", 323),
        ("target_cells_completed_before_extension", 71),
        ("planned_api_cells", 251),
    ),
)
def test_hosted_paired_workflow_rejects_geometry_drift(field: str, value: int) -> None:
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(RuntimeError, match="geometry drift"):
        _assert_geometry(manifest)


def test_hosted_paired_workflow_rejects_identity_imbalance() -> None:
    manifest = _manifest()
    manifest["target_identity_group_counts"] = {"A": 4, "B": 3, "C": 2}
    with pytest.raises(RuntimeError, match="identity balance"):
        _assert_geometry(manifest)


def test_hosted_paired_workflow_rejects_outcome_contaminated_selection() -> None:
    manifest = _manifest()
    manifest["scientific_outcomes_inspected_before_extension_definition"] = True
    with pytest.raises(RuntimeError, match="outcome-blind"):
        _assert_geometry(manifest)


def test_hosted_paired_workflow_rejects_cost_projection_above_target() -> None:
    manifest = _manifest()
    manifest["projected_incremental_cost_with_safety_rmb"] = 190.01
    with pytest.raises(RuntimeError, match="exceeds 190 RMB"):
        _assert_geometry(manifest)
