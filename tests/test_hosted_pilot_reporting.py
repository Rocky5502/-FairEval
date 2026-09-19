from __future__ import annotations

import json
from pathlib import Path

from scripts.render_hosted_pilot_operational import _load, render_table


ROOT = Path(__file__).resolve().parents[1]


def test_hosted_pilot_summary_is_outcome_blind_and_renderable() -> None:
    path = ROOT / "provenance" / "hosted_fairsynth_pilot99_operational_summary.json"
    payload = _load(path)
    assert payload["completed_cells"] == 99
    assert payload["planned_cells"] == 1080
    assert payload["scientific_outcomes_inspected"] is False
    assert payload["paper_use"] == "operational_feasibility_and_cost_provenance_only"

    table = render_table(payload)
    assert "operational pilot" in table.lower()
    assert "99/1,080" in table
    assert "OpenAI & 17 & 3 & 3 & 2 & 6.5473" in table
    assert "Google & 17 & 3 & 3 & 2 & 36.3437" in table
    assert "fairness or utility outcomes" in table


def test_hosted_pilot_summary_has_all_six_families() -> None:
    payload = json.loads(
        (ROOT / "provenance" / "hosted_fairsynth_pilot99_operational_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(payload["families"]) == {
        "openai",
        "anthropic",
        "google",
        "deepseek",
        "qwen",
        "meta",
    }
