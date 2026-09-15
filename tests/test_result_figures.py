import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from scripts.build_result_figures import (
    build_rq1_quadrant,
    build_rq2_forest,
    build_rq3_variance,
    build_rq4_pareto,
)


def _write_jsonl(path: Path, rows: list[dict]):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_rq1_quadrant_writes_vector_pdf(tmp_path: Path):
    source = tmp_path / "rq1_pairs.jsonl"
    _write_jsonl(
        source,
        [
            {
                "rq": "RQ1",
                "contrast": "observed_vs_demographic_counterfactual",
                "model_family": "openai",
                "mean_one_minus_rbo": 0.2,
                "delta_ndcg": 0.1,
            },
            {
                "rq": "RQ1",
                "contrast": "observed_vs_demographic_counterfactual",
                "model_family": "anthropic",
                "mean_one_minus_rbo": 0.4,
                "delta_ndcg": -0.05,
            },
        ],
    )
    output = tmp_path / "rq1.pdf"
    build_rq1_quadrant(source, output)
    data = output.read_bytes()
    assert data.startswith(b"%PDF-")
    assert len(data) > 3000


def test_build_rq1_rejects_only_invalid_behavioral_pairs(tmp_path: Path):
    source = tmp_path / "rq1_pairs.jsonl"
    _write_jsonl(
        source,
        [
            {
                "rq": "RQ1",
                "contrast": "observed_vs_demographic_counterfactual",
                "model_family": "openai",
                "mean_one_minus_rbo": None,
                "delta_ndcg": 0.0,
            }
        ],
    )
    with pytest.raises(ValueError, match="no valid matched-ranking diagnostics"):
        build_rq1_quadrant(source, tmp_path / "rq1.pdf")


def test_build_rq2_forest_writes_vector_pdf(tmp_path: Path):
    source = tmp_path / "inference.jsonl"
    _write_jsonl(
        source,
        [
            {
                "rq": "RQ2",
                "contrast": "true_vs_shuffled_personality",
                "metric": "ndcg",
                "dataset": "personality2018",
                "model_family": "openai",
                "mean_paired_difference": 0.08,
                "bootstrap_ci_low": 0.02,
                "bootstrap_ci_high": 0.14,
            },
            {
                "rq": "RQ2",
                "contrast": "true_vs_shuffled_personality",
                "metric": "ndcg",
                "dataset": "reasoner",
                "model_family": "google",
                "mean_paired_difference": -0.02,
                "bootstrap_ci_low": -0.08,
                "bootstrap_ci_high": 0.04,
            },
        ],
    )
    output = tmp_path / "rq2.pdf"
    build_rq2_forest(source, output)
    data = output.read_bytes()
    assert data.startswith(b"%PDF-")
    assert len(data) > 3000


def test_build_rq3_variance_writes_vector_pdf(tmp_path: Path):
    source = tmp_path / "rq3_summary.jsonl"
    rows = []
    for factor, value in [
        ("prompt", 0.04),
        ("cue", 0.06),
        ("candidate_order", 0.03),
        ("cutoff", 0.08),
        ("stochasticity", 0.02),
    ]:
        rows.append(
            {
                "schema_version": "faireval-rq3-variation-summary-v1",
                "factor": factor,
                "dataset": "movielens_1m",
                "model_family": "openai",
                "requested_model_id": "gpt-5.6-terra",
                "condition_id": "C1",
                "metric": "ndcg",
                "n_users": 10,
                "mean_within_user_sd": value,
                "median_within_user_sd": value,
                "mean_within_user_range": value * 2,
                "median_within_user_range": value * 2,
            }
        )
    _write_jsonl(source, rows)
    output = tmp_path / "rq3.pdf"
    build_rq3_variance(source, output)
    data = output.read_bytes()
    assert data.startswith(b"%PDF-")
    assert len(data) > 3000


def _rq4_artifact() -> dict:
    return {
        "schema_version": "faireval-rq4-pair-artifact-v1",
        "selection_used_test_outcomes": False,
        "per_model_or_dataset_tuning": False,
        "validation_frontier": [
            {
                "alpha": 0.25,
                "lambda_instability": 0.0,
                "pair_identity_ndcg_mean_system": 0.61,
                "utility_retention": 0.98,
                "pair_abs_cug_ndcg_on_available": 0.07,
            },
            {
                "alpha": 0.50,
                "lambda_instability": 0.20,
                "pair_identity_ndcg_mean_system": 0.59,
                "utility_retention": 0.96,
                "pair_abs_cug_ndcg_on_available": 0.03,
            },
            {
                "alpha": 0.75,
                "lambda_instability": 0.80,
                "pair_identity_ndcg_mean_system": 0.55,
                "utility_retention": 0.90,
                "pair_abs_cug_ndcg_on_available": 0.015,
            },
        ],
        "operating_point": {
            "alpha": 0.50,
            "lambda_instability": 0.20,
            "utility_floor_ratio": 0.95,
            "validation_summary": {
                "pair_identity_ndcg_mean_system": 0.59,
                "pair_abs_cug_ndcg_on_available": 0.03,
            },
        },
        "test_summary": {
            "pair_identity_ndcg_mean_system": 0.58,
            "pair_abs_cug_ndcg_on_available": 0.035,
        },
    }


def test_build_rq4_pareto_writes_vector_pdf(tmp_path: Path):
    source = tmp_path / "rq4_pair_artifact.json"
    source.write_text(json.dumps(_rq4_artifact(), indent=2), encoding="utf-8")
    output = tmp_path / "rq4.pdf"
    build_rq4_pareto(source, output)
    data = output.read_bytes()
    assert data.startswith(b"%PDF-")
    assert len(data) > 3000


def test_build_rq4_rejects_test_selected_artifact(tmp_path: Path):
    artifact = _rq4_artifact()
    artifact["selection_used_test_outcomes"] = True
    source = tmp_path / "rq4_bad.json"
    source.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="test outcomes influenced selection"):
        build_rq4_pareto(source, tmp_path / "rq4.pdf")
