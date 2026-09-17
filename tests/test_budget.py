from __future__ import annotations

import json
from pathlib import Path

import pytest

from faireval.budget import BudgetExceeded, ZhizengzengBudgetGuard


class FakeGuard(ZhizengzengBudgetGuard):
    def __init__(self, balances: list[float], *, ledger_path: Path, **kwargs):
        super().__init__(ledger_path=ledger_path, **kwargs)
        self._balances = iter(balances)

    def _query_balance(self) -> float:
        return float(next(self._balances))

    def _api_key(self) -> str:
        return "never-used"


def test_budget_ledger_reconciles_balance_delta(tmp_path: Path) -> None:
    guard = FakeGuard(
        [300.0, 298.5, 297.0],
        ledger_path=tmp_path / "budget.json",
        target_rmb=200,
        hard_cap_rmb=250,
        request_reserve_rmb=2,
    )
    initial = guard.ensure_initialized()
    assert initial.spent_rmb == 0
    guard.before_cell({"cell_id": "abc"}, 0)
    final = guard.snapshot(event="manual")
    assert final.spent_rmb == 3.0
    payload = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert payload["initial_balance_rmb"] == 300.0
    assert payload["latest"]["spent_rmb"] == 3.0


def test_budget_guard_stops_before_cell_near_200_rmb_target(tmp_path: Path) -> None:
    guard = FakeGuard(
        [300.0, 101.5],  # 198.5 RMB spent; only 1.5 RMB remains to target.
        ledger_path=tmp_path / "budget.json",
        target_rmb=200,
        hard_cap_rmb=250,
        request_reserve_rmb=2,
    )
    guard.ensure_initialized()
    with pytest.raises(BudgetExceeded, match="primary budget target"):
        guard.before_cell({"cell_id": "stop"}, 10)


def test_budget_guard_stops_after_cell_at_target(tmp_path: Path) -> None:
    guard = FakeGuard(
        [300.0, 250.0, 99.0],
        ledger_path=tmp_path / "budget.json",
        target_rmb=200,
        hard_cap_rmb=250,
        request_reserve_rmb=2,
    )
    guard.ensure_initialized()
    guard.before_cell({"cell_id": "last"}, 0)
    with pytest.raises(BudgetExceeded, match="reached the primary budget target"):
        guard.after_cell({"cell_id": "last"}, 1)


def test_budget_policy_drift_requires_new_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "budget.json"
    first = FakeGuard(
        [300.0],
        ledger_path=ledger,
        target_rmb=200,
        hard_cap_rmb=250,
        request_reserve_rmb=2,
    )
    first.ensure_initialized()

    second = FakeGuard(
        [300.0],
        ledger_path=ledger,
        target_rmb=190,
        hard_cap_rmb=250,
        request_reserve_rmb=2,
    )
    with pytest.raises(RuntimeError, match="Budget policy drift"):
        second.ensure_initialized()
