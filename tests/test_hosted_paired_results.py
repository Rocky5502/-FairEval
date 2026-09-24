from __future__ import annotations

from pathlib import Path

from scripts.render_hosted_paired_results import (
    FAMILY_ORDER,
    ID_CONTRAST,
    P_CONTRAST,
    render_figure,
    render_summary,
    render_table,
)


def _inference_rows():
    rows = []
    for family_i, family in enumerate(FAMILY_ORDER):
        for contrast_i, (rq, contrast) in enumerate(
            (
                ("SYNTH-ID", ID_CONTRAST),
                ("SYNTH-PERSONALITY", P_CONTRAST),
            )
        ):
            for metric in ("ndcg", "recall"):
                mean = (family_i - 2.5) * 0.005 + contrast_i * 0.002
                rows.append(
                    {
                        "rq": rq,
                        "contrast": contrast,
                        "metric": metric,
                        "dataset": "fairsynth360",
                        "model_family": family,
                        "requested_model_id": family,
                        "n_users": 9,
                        "mean_paired_difference": mean,
                        "bootstrap_ci_low": mean - 0.02,
                        "bootstrap_ci_high": mean + 0.02,
                        "holm_adjusted_p": 0.5,
                    }
                )
    return rows


def _user_condition_rows():
    rows = []
    conditions = (
        ("C0", 0.31),
        ("C1", 0.32),
        ("C2:synthetic_identity_group_A_to_B", 0.30),
        ("C2:synthetic_identity_group_A_to_C", 0.31),
        ("C3", 0.33),
        ("C4", 0.32),
    )
    for family_i, family in enumerate(FAMILY_ORDER):
        for user_i in range(9):
            for condition_id, base in conditions:
                rows.append(
                    {
                        "model_family": family,
                        "user_id": f"u{user_i:03d}",
                        "condition_id": condition_id,
                        "ndcg": base + family_i * 0.002,
                    }
                )
    return rows


def test_hosted_paired_table_and_summary_are_complete():
    rows = _inference_rows()
    table = render_table(rows)
    summary = render_summary(rows)
    for family in ("OpenAI", "Anthropic", "Google", "DeepSeek", "Qwen", "Llama"):
        assert family in table
    assert "Identity" in table
    assert "Personality" in table
    assert "$N=9$" in summary
    assert "real-world demographic fairness" in summary


def test_hosted_profile_figure_renders_nonempty_pdf(tmp_path: Path):
    output = tmp_path / "hosted_profiles.pdf"
    render_figure(_user_condition_rows(), output)
    assert output.is_file()
    assert output.stat().st_size > 0
