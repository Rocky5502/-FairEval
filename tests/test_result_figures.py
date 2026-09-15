import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from scripts.build_result_figures import build_rq1_quadrant, build_rq2_forest


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
