from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_GATEWAY_BASE_URL = "https://api.zhizengzeng.com/v1"


class BudgetExceeded(RuntimeError):
    """Raised when the hosted run must stop under the frozen RMB budget policy."""


@dataclass(frozen=True)
class BudgetSnapshot:
    available_rmb: float
    spent_rmb: float
    target_rmb: float
    hard_cap_rmb: float
    initial_balance_rmb: float
    checked_unix: float


class ZhizengzengBudgetGuard:
    """Balance-reconciled client-side RMB budget guard for FairEval hosted execution.

    The gateway documents a credit endpoint that returns ``available_amount``.
    We record an initial balance and reconcile experiment spend as balance
    movement. The preferred stop is the 200 RMB target: no new cell is launched
    when the remaining target budget is at or below the frozen request reserve.
    The separate 250 RMB value is an emergency client-side stop threshold that
    retains a large safety buffer rather than being approached normally.

    FairEval does not claim that this threshold is an atomic provider-side spend
    cap. A provider could in principle charge an unexpectedly large in-flight
    request before the next balance reconciliation. The protocol therefore stops
    around 200 RMB, retains roughly 50 RMB of emergency headroom, keeps outputs
    short, and checks balance before and after every persisted cell.

    A cell can contain one format-only repair call, so ``request_reserve_rmb`` is
    held back before the cell starts. The reserve protects the normal-stop target;
    the larger target-to-threshold gap is an additional operational safety buffer.
    """

    def __init__(
        self,
        *,
        ledger_path: Path,
        api_key_env: str = "ZZZ_API_KEY",
        base_url: str | None = None,
        target_rmb: float = 200.0,
        hard_cap_rmb: float = 250.0,
        request_reserve_rmb: float = 2.0,
    ) -> None:
        if target_rmb <= 0 or hard_cap_rmb <= 0:
            raise ValueError("budget values must be positive")
        if target_rmb >= hard_cap_rmb:
            raise ValueError("target_rmb must be strictly below hard_cap_rmb")
        if request_reserve_rmb <= 0 or request_reserve_rmb >= target_rmb:
            raise ValueError("request_reserve_rmb must be positive and below target_rmb")
        self.ledger_path = ledger_path
        self.api_key_env = api_key_env
        self.base_url = (
            base_url or os.environ.get("ZZZ_BASE_URL") or DEFAULT_GATEWAY_BASE_URL
        ).rstrip("/")
        self.target_rmb = float(target_rmb)
        self.hard_cap_rmb = float(hard_cap_rmb)
        self.request_reserve_rmb = float(request_reserve_rmb)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"Missing environment variable {self.api_key_env}")
        return key

    def _query_balance(self) -> float:
        req = urllib.request.Request(
            f"{self.base_url}/dashboard/billing/credit_grants",
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "FairEval-ECIR2027-budget-guard/1.0",
            },
            data=b"{}",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
        payload = json.loads(raw.decode("utf-8"))
        try:
            return float(payload["grants"]["available_amount"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Unexpected Zhizengzeng balance response schema") from exc

    def _load_ledger(self) -> dict[str, Any] | None:
        if not self.ledger_path.exists():
            return None
        payload = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("budget ledger must contain a JSON object")
        return payload

    def _write_ledger(self, payload: Mapping[str, Any]) -> None:
        tmp = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self.ledger_path)

    def ensure_initialized(self) -> BudgetSnapshot:
        ledger = self._load_ledger()
        available = self._query_balance()
        if ledger is None:
            ledger = {
                "schema_version": "faireval-hosted-budget-ledger-v1",
                "gateway": "zhizengzeng",
                "base_url": self.base_url,
                "api_key_env": self.api_key_env,
                "initial_balance_rmb": available,
                "target_rmb": self.target_rmb,
                "hard_cap_rmb": self.hard_cap_rmb,
                "hard_cap_semantics": "client_side_emergency_stop_threshold",
                "provider_side_atomic_spend_cap_claimed": False,
                "request_reserve_rmb": self.request_reserve_rmb,
                "checks": [],
            }
        else:
            frozen = (
                float(ledger.get("target_rmb", -1)),
                float(ledger.get("hard_cap_rmb", -1)),
                float(ledger.get("request_reserve_rmb", -1)),
            )
            current = (self.target_rmb, self.hard_cap_rmb, self.request_reserve_rmb)
            if frozen != current:
                raise RuntimeError(
                    "Budget policy drift detected. Use a new ledger path rather than changing "
                    "target/hard-cap/reserve in an existing experiment ledger."
                )
        return self._append_check(ledger, available=available, event="initialize_or_resume")

    def _append_check(
        self,
        ledger: dict[str, Any],
        *,
        available: float,
        event: str,
        cell_id: str | None = None,
        executed_cells: int | None = None,
    ) -> BudgetSnapshot:
        initial = float(ledger["initial_balance_rmb"])
        spent = max(0.0, initial - available)
        checked = time.time()
        record = {
            "event": event,
            "checked_unix": checked,
            "available_rmb": round(available, 6),
            "spent_rmb": round(spent, 6),
            "cell_id": cell_id,
            "executed_cells": executed_cells,
        }
        checks = ledger.setdefault("checks", [])
        if not isinstance(checks, list):
            raise RuntimeError("budget ledger checks must be a list")
        checks.append(record)
        ledger["latest"] = record
        ledger["target_reached"] = spent >= self.target_rmb
        ledger["emergency_threshold_reached"] = spent >= self.hard_cap_rmb
        self._write_ledger(ledger)
        return BudgetSnapshot(
            available_rmb=available,
            spent_rmb=spent,
            target_rmb=self.target_rmb,
            hard_cap_rmb=self.hard_cap_rmb,
            initial_balance_rmb=initial,
            checked_unix=checked,
        )

    def snapshot(
        self,
        *,
        event: str,
        cell_id: str | None = None,
        executed_cells: int | None = None,
    ) -> BudgetSnapshot:
        ledger = self._load_ledger()
        if ledger is None:
            return self.ensure_initialized()
        available = self._query_balance()
        return self._append_check(
            ledger,
            available=available,
            event=event,
            cell_id=cell_id,
            executed_cells=executed_cells,
        )

    def before_cell(self, row: Mapping[str, Any], executed_cells: int) -> None:
        snap = self.snapshot(
            event="before_cell",
            cell_id=str(row.get("cell_id", "")) or None,
            executed_cells=executed_cells,
        )
        if snap.spent_rmb >= self.hard_cap_rmb:
            raise BudgetExceeded(
                f"Hosted execution cannot continue: spent={snap.spent_rmb:.4f} RMB "
                f"already reaches emergency threshold={self.hard_cap_rmb:.2f} RMB."
            )

        remaining_to_target = self.target_rmb - snap.spent_rmb
        if remaining_to_target <= self.request_reserve_rmb:
            raise BudgetExceeded(
                "Hosted execution stopped before the next cell at the primary budget target: "
                f"spent={snap.spent_rmb:.4f} RMB, target={self.target_rmb:.2f} RMB, "
                f"reserve={self.request_reserve_rmb:.2f} RMB, "
                f"emergency_threshold={self.hard_cap_rmb:.2f} RMB."
            )

    def after_cell(self, row: Mapping[str, Any], executed_cells: int) -> None:
        snap = self.snapshot(
            event="after_cell",
            cell_id=str(row.get("cell_id", "")) or None,
            executed_cells=executed_cells,
        )
        if snap.spent_rmb >= self.hard_cap_rmb:
            raise BudgetExceeded(
                "Hosted execution reached the emergency client-side stop threshold after a "
                f"persisted cell: spent={snap.spent_rmb:.4f} RMB >= {self.hard_cap_rmb:.2f} RMB."
            )
        if snap.spent_rmb >= self.target_rmb:
            raise BudgetExceeded(
                "Hosted execution reached the primary budget target after a persisted cell; "
                f"spent={snap.spent_rmb:.4f} RMB >= {self.target_rmb:.2f} RMB. "
                f"The {self.hard_cap_rmb:.2f} RMB emergency threshold remains safety headroom."
            )

    def report(self) -> dict[str, Any]:
        snap = self.snapshot(event="report")
        return {
            "schema_version": "faireval-hosted-budget-report-v2",
            **asdict(snap),
            "hard_cap_semantics": "client_side_emergency_stop_threshold",
            "provider_side_atomic_spend_cap_claimed": False,
            "remaining_to_target_rmb": max(0.0, self.target_rmb - snap.spent_rmb),
            "remaining_to_emergency_threshold_rmb": max(
                0.0, self.hard_cap_rmb - snap.spent_rmb
            ),
            "request_reserve_rmb": self.request_reserve_rmb,
            "ledger_path": str(self.ledger_path),
        }
