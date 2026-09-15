from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs"


def _load(name: str):
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))


def test_analysis_policy_matches_preregistered_statistics():
    analysis = _load("analysis.yaml")
    study = _load("study_design.yaml")

    assert analysis["invalid_outputs"]["primary_system_utility"] == "zero"
    assert analysis["invalid_outputs"]["complete_case_only_primary_analysis"] is False
    assert analysis["invalid_outputs"]["report_invalid_output_disparity"] is True

    assert analysis["inference"]["confidence"] == study["statistics"]["confidence_level"]
    assert analysis["inference"]["bootstrap_samples"] == study["statistics"]["bootstrap_samples"]
    assert analysis["inference"]["multiple_comparisons"]["method"] == "holm"
    assert study["statistics"]["paired_permutation_primary"] is True
    assert study["statistics"]["wilcoxon_sensitivity"] is True


def test_analysis_correction_family_is_frozen_before_results():
    analysis = _load("analysis.yaml")
    correction = analysis["inference"]["multiple_comparisons"]
    assert correction["family"] == "rq_metric_contrast"
    assert analysis["status"] == "pre_run"
