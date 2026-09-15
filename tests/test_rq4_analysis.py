import pytest

from faireval.rq4_analysis import select_operating_point


def test_select_operating_point_respects_utility_floor_and_minimizes_gap():
    rows = [
        {
            "alpha": 0.25,
            "lambda_instability": 0.0,
            "utility_retention": 1.00,
            "pair_identity_ndcg_mean_system": 0.60,
            "pair_abs_cug_ndcg_on_available": 0.08,
        },
        {
            "alpha": 0.50,
            "lambda_instability": 0.20,
            "utility_retention": 0.96,
            "pair_identity_ndcg_mean_system": 0.58,
            "pair_abs_cug_ndcg_on_available": 0.03,
        },
        {
            "alpha": 0.75,
            "lambda_instability": 0.80,
            "utility_retention": 0.90,
            "pair_identity_ndcg_mean_system": 0.54,
            "pair_abs_cug_ndcg_on_available": 0.01,
        },
    ]
    selected = select_operating_point(rows, utility_floor_ratio=0.95)
    assert selected["alpha"] == 0.50
    assert selected["lambda_instability"] == 0.20
    assert selected["selection_objective"] == "min_abs_cug_subject_to_validation_utility_retention"


def test_select_operating_point_deterministic_tie_break():
    rows = [
        {
            "alpha": 0.50,
            "lambda_instability": 0.40,
            "utility_retention": 0.98,
            "pair_identity_ndcg_mean_system": 0.60,
            "pair_abs_cug_ndcg_on_available": 0.02,
        },
        {
            "alpha": 0.25,
            "lambda_instability": 0.20,
            "utility_retention": 0.98,
            "pair_identity_ndcg_mean_system": 0.60,
            "pair_abs_cug_ndcg_on_available": 0.02,
        },
    ]
    selected = select_operating_point(rows, utility_floor_ratio=0.95)
    assert selected["lambda_instability"] == 0.20
    assert selected["alpha"] == 0.25


def test_select_operating_point_refuses_post_hoc_floor_relaxation():
    rows = [
        {
            "alpha": 0.50,
            "lambda_instability": 0.20,
            "utility_retention": 0.94,
            "pair_identity_ndcg_mean_system": 0.57,
            "pair_abs_cug_ndcg_on_available": 0.01,
        }
    ]
    with pytest.raises(ValueError, match="no PAIR grid point satisfies"):
        select_operating_point(rows, utility_floor_ratio=0.95)
