from __future__ import annotations

from pathlib import Path

import yaml

from faireval.datasets.factory import DATASET_IDS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs"


def _load(name: str):
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))


def _require_known_datasets(values, *, where: str) -> None:
    unknown = sorted(set(values) - set(DATASET_IDS))
    if unknown:
        raise SystemExit(f"{where} contains unknown dataset IDs: {unknown}")


def main() -> int:
    datasets = _load("datasets.yaml")
    study = _load("study_design.yaml")
    experiment = _load("experiment.yaml")
    counterfactuals = _load("counterfactuals.yaml")
    cues = _load("cue_suite.yaml")
    prompts = _load("prompt_suite.yaml")
    models = _load("models.yaml")

    manifest_ids = set(datasets["datasets"])
    executable_ids = set(DATASET_IDS)
    if manifest_ids != executable_ids:
        raise SystemExit(
            "dataset manifest/factory mismatch: "
            f"manifest_only={sorted(manifest_ids - executable_ids)}, "
            f"factory_only={sorted(executable_ids - manifest_ids)}"
        )
    if "lastfm" in manifest_ids or "lastfm_1k" not in manifest_ids:
        raise SystemExit("Last.fm must use the executable adapter ID lastfm_1k")

    _require_known_datasets(
        study["rq1_demographic_counterfactual_fairness"]["datasets"],
        where="study_design.rq1",
    )
    _require_known_datasets(
        study["rq2_grounded_personality_value"]["datasets"],
        where="study_design.rq2",
    )
    _require_known_datasets(
        study["rq3_reliability"]["cue_style_robustness"]["datasets"],
        where="study_design.rq3.cues",
    )
    _require_known_datasets(study["rq4_mitigation"]["datasets"], where="study_design.rq4")

    experiment_datasets: set[str] = set()
    for track_name, track in experiment["tracks"].items():
        _require_known_datasets(track["datasets"], where=f"experiment.tracks.{track_name}")
        experiment_datasets.update(str(value) for value in track["datasets"])
    if experiment_datasets != executable_ids:
        raise SystemExit(
            "experiment tracks must cover each executable dataset exactly by ID: "
            f"got={sorted(experiment_datasets)} expected={sorted(executable_ids)}"
        )
    _require_known_datasets(experiment["rq4"]["demographic_datasets"], where="experiment.rq4")

    if datasets["datasets"]["mind"].get("core_conditions") != ["preference_only"]:
        raise SystemExit("datasets.yaml must keep MIND core scope preference-only")
    if study["mind_scope"].get("core_conditions") != ["preference_only"]:
        raise SystemExit("study_design.yaml must keep MIND core scope preference-only")
    mind_experiment = experiment["tracks"]["generalization_only"]
    if mind_experiment.get("core_conditions") != ["C0"]:
        raise SystemExit("experiment.yaml must keep MIND core condition C0 only")
    if mind_experiment.get("synthetic_identity_in_core_plan") is not False:
        raise SystemExit("experiment.yaml must not enable synthetic MIND identity cells in core")
    mind_stress = counterfactuals["mind"]["synthetic_stress_test"]
    if mind_stress.get("enabled_in_core_plan") is not False:
        raise SystemExit("counterfactuals.yaml must keep synthetic MIND identity outside core")

    c6 = next(row for row in experiment["conditions"] if row["id"] == "C6")
    if c6.get("enabled_in_core_plan") is not False:
        raise SystemExit("C6 must remain disabled until the executable planner implements it")

    cue_rows = cues["cue_suite"]["robustness_subset"]["pre_registered_cues"]
    cue_ids = {row["cue_id"] for row in cue_rows}
    primary_cue = cues["cue_suite"]["primary"]["cue_id"]
    if primary_cue not in cue_ids:
        raise SystemExit(f"primary cue {primary_cue!r} is not in the registered cue suite")
    if study["primary_prompt"]["cue_id"] != primary_cue:
        raise SystemExit("study_design primary cue disagrees with cue_suite")
    if experiment["prompt_design"]["primary_cue"] != primary_cue:
        raise SystemExit("experiment primary cue disagrees with cue_suite")

    registered_robustness_cues = set(study["rq3_reliability"]["cue_style_robustness"]["cue_ids"])
    if registered_robustness_cues != cue_ids:
        raise SystemExit(
            "study_design cue IDs disagree with cue_suite: "
            f"study={sorted(registered_robustness_cues)}, suite={sorted(cue_ids)}"
        )

    prompt_suite = prompts["prompt_suite"]
    primary_template = prompt_suite["primary"]["template_id"]
    if study["primary_prompt"]["template_id"] != primary_template:
        raise SystemExit("study_design primary template disagrees with prompt_suite")
    if experiment["prompt_design"]["primary_template"] != primary_template:
        raise SystemExit("experiment primary template disagrees with prompt_suite")
    if primary_template not in prompt_suite["paraphrase_robustness"]["templates"]:
        raise SystemExit("primary template is missing from the registered paraphrase suite")

    enabled_models = [row for row in models["models"] if row.get("enabled", True)]
    if len(enabled_models) != 6:
        raise SystemExit(f"expected six enabled core model families, found {len(enabled_models)}")
    families = {row["family"] for row in enabled_models}
    if len(families) != 6:
        raise SystemExit("enabled model families must be unique")
    for row in enabled_models:
        if "reasoning_or_thinking_setting" not in row:
            raise SystemExit(f"model {row['family']} lacks reasoning_or_thinking_setting")
        if "sampling_policy" not in row:
            raise SystemExit(f"model {row['family']} lacks sampling_policy")

    primary_generation = study["primary_generation"]
    model_generation = models["generation"]
    if model_generation["top_k"] != study["primary_prompt"]["k"]:
        raise SystemExit("models.yaml top_k disagrees with study_design primary K")
    if model_generation["main_repetitions"] != primary_generation["repetitions"]:
        raise SystemExit("models.yaml repetition count disagrees with study_design")
    if model_generation["temperature_main_requested"] != primary_generation["temperature_requested"]:
        raise SystemExit("models.yaml requested temperature disagrees with study_design")
    if model_generation["top_p_main_requested"] != primary_generation["top_p_requested"]:
        raise SystemExit("models.yaml requested top_p disagrees with study_design")

    expected_rq1 = {"movielens_1m", "lastfm_1k"}
    if set(study["rq1_demographic_counterfactual_fairness"]["datasets"]) != expected_rq1:
        raise SystemExit("RQ1 observed-demographic dataset scope drifted")
    if set(study["rq4_mitigation"]["datasets"]) != expected_rq1:
        raise SystemExit("RQ4 demographic dataset scope drifted")

    print("config preflight: six executable dataset IDs match datasets.yaml")
    print("config preflight: RQ1/RQ4/MIND scopes match the executable core plan")
    print("config preflight: primary prompt/cue IDs agree across YAML manifests")
    print("config preflight: generation K/repetitions/sampling requests agree")
    print("config preflight: six model families include frozen deliberation/sampling policies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
