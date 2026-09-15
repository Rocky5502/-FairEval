from pathlib import Path

import yaml

from faireval.datasets.factory import DATASET_IDS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs"


def _yaml(name: str):
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))


def _track_datasets(experiment: dict) -> set[str]:
    datasets: set[str] = set()
    for track in experiment["tracks"].values():
        value = track.get("datasets", [])
        if isinstance(value, list):
            datasets.update(str(x) for x in value)
    return datasets


def test_dataset_manifest_matches_executable_adapter_ids():
    manifest = _yaml("datasets.yaml")
    assert set(manifest["datasets"]) == set(DATASET_IDS)
    assert "lastfm" not in manifest["datasets"]
    assert "lastfm_1k" in manifest["datasets"]


def test_experiment_tracks_cover_exactly_the_six_adapter_ids():
    experiment = _yaml("experiment.yaml")
    assert _track_datasets(experiment) == set(DATASET_IDS)


def test_study_design_uses_executable_dataset_ids():
    study = _yaml("study_design.yaml")
    named_lists = [
        study["rq1_demographic_counterfactual_fairness"]["datasets"],
        study["rq2_grounded_personality_value"]["datasets"],
        study["rq3_reliability"]["cue_style_robustness"]["datasets"],
        study["rq4_mitigation"]["datasets"],
    ]
    flattened = {str(x) for values in named_lists for x in values}
    assert flattened <= set(DATASET_IDS)
    assert "lastfm" not in flattened


def test_mind_has_no_identity_cells_in_frozen_core_configs():
    experiment = _yaml("experiment.yaml")
    study = _yaml("study_design.yaml")
    counterfactuals = _yaml("counterfactuals.yaml")

    assert experiment["tracks"]["generalization_only"]["core_conditions"] == ["C0"]
    assert experiment["tracks"]["generalization_only"]["synthetic_identity_in_core_plan"] is False
    assert study["mind_scope"]["core_conditions"] == ["preference_only"]
    assert study["mind_scope"]["synthetic_identity_stress_test"]["enabled_in_core_plan"] is False
    assert counterfactuals["mind"]["synthetic_stress_test"]["enabled_in_core_plan"] is False


def test_primary_prompt_and_cue_ids_agree_across_configs():
    experiment = _yaml("experiment.yaml")
    study = _yaml("study_design.yaml")
    prompt_suite = _yaml("prompt_suite.yaml")["prompt_suite"]
    cue_suite = _yaml("cue_suite.yaml")["cue_suite"]

    primary_template = experiment["prompt_design"]["primary_template"]
    assert study["primary_prompt"]["template_id"] == primary_template
    assert prompt_suite["primary"]["template_id"] == primary_template
    assert primary_template in prompt_suite["paraphrase_robustness"]["templates"]

    primary_cue = experiment["prompt_design"]["primary_cue"]
    assert study["primary_prompt"]["cue_id"] == primary_cue
    assert cue_suite["primary"]["cue_id"] == primary_cue


def test_primary_generation_contract_is_consistent():
    experiment = _yaml("experiment.yaml")
    study = _yaml("study_design.yaml")
    models = _yaml("models.yaml")
    prompt_suite = _yaml("prompt_suite.yaml")["prompt_suite"]

    assert experiment["study"]["primary_k"] == study["primary_prompt"]["k"]
    assert prompt_suite["primary"]["k"] == study["primary_prompt"]["k"]
    assert experiment["study"]["main_repetitions"] == study["primary_generation"]["repetitions"]
    assert prompt_suite["primary"]["repetitions"] == study["primary_generation"]["repetitions"]
    assert models["generation"]["top_k"] == study["primary_prompt"]["k"]
    assert models["generation"]["main_repetitions"] == study["primary_generation"]["repetitions"]
    assert models["generation"]["temperature_main_requested"] == study["primary_generation"]["temperature_requested"]
    assert models["generation"]["top_p_main_requested"] == study["primary_generation"]["top_p_requested"]
