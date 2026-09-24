from __future__ import annotations

import pytest

from faireval.budget import BudgetExceeded
from scripts.run_hosted_budgeted import _enforce_first_launch_balance


def test_first_launch_balance_blocks_underfunded_new_ledger():
    with pytest.raises(BudgetExceeded, match="below required minimum"):
        _enforce_first_launch_balance(
            available_rmb=199.99,
            minimum_initial_balance_rmb=200.0,
            ledger_preexisting=False,
        )


def test_first_launch_balance_accepts_funded_new_ledger():
    _enforce_first_launch_balance(
        available_rmb=200.0,
        minimum_initial_balance_rmb=200.0,
        ledger_preexisting=False,
    )


def test_resume_does_not_reapply_initial_balance_gate():
    _enforce_first_launch_balance(
        available_rmb=80.0,
        minimum_initial_balance_rmb=200.0,
        ledger_preexisting=True,
    )
