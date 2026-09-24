from scripts.analyze_v7_secondary import _collapse_counterfactuals, _repetition_stability


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
    assert c2["ndcg"] == 0.3
    assert c2["recall"] == 0.4


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
