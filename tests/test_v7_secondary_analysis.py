import json
import sys

import pytest
from scripts.analyze_v7_secondary import (
    _collapse_counterfactuals,
    _render_figure,
    _repetition_stability,
    main,
)


def _row(condition_id: str, *, ndcg: float, recall: float, invalid_rate: float = 0.0):
    return {
        "model_family": "phi35_local",
        "user_id": "u001",
        "condition_id": condition_id,
        "ndcg": ndcg,
        "recall": recall,
        "mrr": ndcg,
        "invalid_rate": invalid_rate,
        "k": 10,
        "ranking_by_repetition": [
            {"repetition": 0, "valid": True, "ranking": [f"i{x:03d}" for x in range(10)]},
            {"repetition": 1, "valid": True, "ranking": [f"i{x:03d}" for x in range(10)]},
        ],
    }


def test_counterfactual_identity_is_collapsed_within_user_before_summary():
    rows = [
        _row("C0", ndcg=0.5, recall=0.6),
        _row("C2:A", ndcg=0.2, recall=0.3),
        _row("C2:B", ndcg=0.4, recall=0.5),
    ]
    collapsed = _collapse_counterfactuals(rows)
    c2 = next(row for row in collapsed if row["condition_group"] == "counterfactual_identity")
    assert c2["n_source_conditions"] == 2
    assert c2["ndcg"] == pytest.approx(0.3)
    assert c2["recall"] == pytest.approx(0.4)


def test_repeat_stability_uses_only_valid_two_repetition_cells():
    rows = [_row("C0", ndcg=0.5, recall=0.6)]
    # Add the second model so the helper's fixed reporting order is satisfied.
    qwen = _row("C0", ndcg=0.4, recall=0.5)
    qwen["model_family"] = "qwen25_local"
    rows.append(qwen)

    summary = _repetition_stability(rows)
    by_model = {row["model_family"]: row for row in summary}
    for model in ("phi35_local", "qwen25_local"):
        row = by_model[model]
        assert row["valid_repeat_pairs"] == 1
        assert row["valid_repeat_pair_rate"] == 1.0
        assert row["mean_rbo_at_10"] == 1.0
        assert row["mean_jaccard_at_10"] == 1.0
        assert row["exact_order_fraction"] == 1.0


def test_secondary_figure_renderer_smoke(tmp_path):
    condition_summary = []
    for model in ("phi35_local", "qwen25_local"):
        for i, condition in enumerate(
            (
                "preference_only",
                "observed_identity",
                "counterfactual_identity",
                "true_ocean",
                "shuffled_ocean",
            )
        ):
            mean = 0.30 + 0.01 * i
            condition_summary.append(
                {
                    "model_family": model,
                    "condition_group": condition,
                    "ndcg_mean": mean,
                    "ndcg_ci_low": mean - 0.02,
                    "ndcg_ci_high": mean + 0.02,
                }
            )

    identity_pairs = [
        {"model_family": model, "delta_ndcg": delta}
        for model in ("phi35_local", "qwen25_local")
        for delta in (-0.02, 0.0, 0.03)
    ]
    personality_pairs = [
        {"model_family": model, "delta_ndcg": delta}
        for model in ("phi35_local", "qwen25_local")
        for delta in (-0.01, 0.0, 0.02)
    ]
    output = tmp_path / "secondary.pdf"
    _render_figure(condition_summary, identity_pairs, personality_pairs, output)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_secondary_main_writes_nonempty_figure_and_table(tmp_path, monkeypatch):
    user_condition = []
    condition_ids = ("C0", "C1", "C2:A", "C3", "C4")
    for model in ("phi35_local", "qwen25_local"):
        for i, condition_id in enumerate(condition_ids):
            row = _row(
                condition_id,
                ndcg=0.25 + 0.01 * i,
                recall=0.40 + 0.01 * i,
            )
            row["model_family"] = model
            user_condition.append(row)

    identity_pairs = [
        {"model_family": "phi35_local", "delta_ndcg": 0.01},
        {"model_family": "qwen25_local", "delta_ndcg": -0.01},
    ]
    personality_pairs = [
        {"model_family": "phi35_local", "delta_ndcg": -0.02},
        {"model_family": "qwen25_local", "delta_ndcg": 0.02},
    ]

    def write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    user_path = tmp_path / "user_condition.jsonl"
    identity_path = tmp_path / "identity_pairs.jsonl"
    personality_path = tmp_path / "personality_pairs.jsonl"
    write_jsonl(user_path, user_condition)
    write_jsonl(identity_path, identity_pairs)
    write_jsonl(personality_path, personality_pairs)

    output_dir = tmp_path / "analysis"
    figure = tmp_path / "secondary.pdf"
    table = tmp_path / "stability.tex"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_v7_secondary.py",
            "--user-condition", str(user_path),
            "--identity-pairs", str(identity_path),
            "--personality-pairs", str(personality_path),
            "--output-dir", str(output_dir),
            "--paper-figure", str(figure),
            "--paper-stability-table", str(table),
            "--bootstrap-samples", "20",
        ],
    )

    assert main() == 0
    assert figure.is_file() and figure.stat().st_size > 0
    assert table.is_file() and table.stat().st_size > 0
    table_text = table.read_text(encoding="utf-8")
    assert "Phi-3.5-mini" in table_text
    assert "Qwen2.5-7B" in table_text
    summary = json.loads((output_dir / "secondary_summary.json").read_text(encoding="utf-8"))
    assert summary["new_model_or_api_calls"] is False
    assert len(summary["repetition_stability_summary"]) == 2
