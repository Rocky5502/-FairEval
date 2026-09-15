from __future__ import annotations

from pathlib import Path

import yaml

from faireval.datasets.factory import AUXILIARY_DATASET_IDS, DATASET_IDS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs"


def _load(name: str):
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))


def _require_known_datasets(values, *, where: str) -> None:
    unknown = sorted(set(values) - set(DATASET_IDS))
    if unknown:
        raise SystemExit(f"{where} contains unknown core dataset IDs: {unknown}")


def main() -> int:
    datasets = _load("datasets.yaml")
    releases = _load("dataset_releases.yaml")
    acquisition = _load("dataset_acquisition.yaml")
    study = _load("study_design.yaml")
    experiment = _load("experiment.yaml")
    counterfactuals = _load("counterfactuals.yaml")
    cues = _load("cue_suite.yaml")
    prompts = _load("prompt_suite.yaml")
    models = _load("models.yaml")
    local_models = _load("local_models.yaml")
    fairsynth = _load("fairsynth360.yaml")

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

    release_ids = set(releases.get("datasets", {}))
    acquisition_ids = set(acquisition.get("datasets", {}))
    if release_ids != executable_ids:
        raise SystemExit(
            "dataset_releases.yaml IDs disagree with executable core adapters: "
            f"release_only={sorted(release_ids - executable_ids)}, "
            f"missing={sorted(executable_ids - release_ids)}"
        )
    if acquisition_ids != executable_ids:
        raise SystemExit(
            "dataset_acquisition.yaml IDs disagree with executable core adapters: "
            f"acquisition_only={sorted(acquisition_ids - executable_ids)}, "
            f"missing={sorted(executable_ids - acquisition_ids)}"
        )
    for dataset_id in sorted(executable_ids):
        spec = acquisition["datasets"][dataset_id]
        if not isinstance(spec, dict) or not spec.get("source"):
            raise SystemExit(f"dataset acquisition spec for {dataset_id} lacks source")
        if not spec.get("required_all") and not spec.get("required_any_of"):
            raise SystemExit(f"dataset acquisition spec for {dataset_id} lacks raw-file contract")

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
            "experiment core tracks must cover each core dataset exactly by ID: "
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
        raise SystemExit("C6 must remain disabled until a separately versioned plan implements it")

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
    if study["primary_prompt"]["candidate_order"] != "deterministic_per_user_frozen_seed":
        raise SystemExit("study_design must freeze a deterministic per-user candidate-order seed")
    if experiment["prompt_design"]["candidate_order_policy"] != "deterministic_per_user_frozen_seed":
        raise SystemExit("experiment candidate-order policy disagrees with study_design")

    enabled_models = [row for row in models["models"] if row.get("enabled", True)]
    if len(enabled_models) != 6:
        raise SystemExit(f"expected six enabled hosted core model families, found {len(enabled_models)}")
    families = {row["family"] for row in enabled_models}
    if len(families) != 6:
        raise SystemExit("enabled hosted model families must be unique")
    for row in enabled_models:
        family = row["family"]
        if "reasoning_or_thinking_setting" not in row:
            raise SystemExit(f"model {family} lacks reasoning_or_thinking_setting")
        if "sampling_policy" not in row:
            raise SystemExit(f"model {family} lacks sampling_policy")
        if row.get("output_token_parameter") not in {
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
        }:
            raise SystemExit(f"model {family} lacks a recognized output_token_parameter")

    by_family = {row["family"]: row for row in enabled_models}
    if by_family["openai"]["output_token_parameter"] != "max_completion_tokens":
        raise SystemExit("OpenAI core config must use current max_completion_tokens")
    if by_family["anthropic"]["output_token_parameter"] != "max_tokens":
        raise SystemExit("Anthropic core config must use max_tokens")
    if by_family["google"]["output_token_parameter"] != "max_output_tokens":
        raise SystemExit("Gemini core config must use max_output_tokens")

    local_enabled = [row for row in local_models["models"] if row.get("enabled", True)]
    local_by_family = {str(row["family"]): row for row in local_enabled}
    if set(local_by_family) != {"qwen25_local", "phi35_local"}:
        raise SystemExit("local model panel must contain exactly qwen25_local and phi35_local")
    expected_local_ids = {
        "qwen25_local": "Qwen/Qwen2.5-7B-Instruct",
        "phi35_local": "microsoft/Phi-3.5-mini-instruct",
    }
    expected_local_licenses = {"qwen25_local": "Apache-2.0", "phi35_local": "MIT"}
    for family, model_id in expected_local_ids.items():
        row = local_by_family[family]
        if row.get("model_id") != model_id:
            raise SystemExit(f"{family} model ID drifted: {row.get('model_id')!r}")
        if row.get("license") != expected_local_licenses[family]:
            raise SystemExit(f"{family} license metadata drifted")
        if row.get("serving_mode") != "direct_transformers":
            raise SystemExit(f"{family} must use the direct Transformers white-box path")
        if row.get("output_token_parameter") != "max_new_tokens":
            raise SystemExit(f"{family} must use max_new_tokens")
        if row.get("revision") != "pin_exact_huggingface_commit_before_pilot":
            raise SystemExit(f"{family} exact revision must remain explicitly pending before freeze")

    hardware = local_models.get("hardware_profile", {}).get("preferred_single_gpu", {})
    if hardware.get("gpu") != "NVIDIA GeForce RTX 5090" or hardware.get("vram_gb") != 32:
        raise SystemExit("local hardware profile must describe the official RTX 5090 32GB target")

    if set(AUXILIARY_DATASET_IDS) != {"fairsynth360"}:
        raise SystemExit("unexpected auxiliary dataset registry drift")
    if fairsynth.get("scope", {}).get("total_users") != 360:
        raise SystemExit("FairSynth-360 total_users must remain 360 for v1")
    identity = fairsynth.get("identity_control", {})
    if identity.get("values") != ["A", "B", "C"] or identity.get("users_per_group") != 120:
        raise SystemExit("FairSynth-360 identity balance must remain A/B/C = 120 each")
    if identity.get("generated_independently_of_relevance") is not True:
        raise SystemExit("FairSynth identity must remain independent of relevance")
    personality = fairsynth.get("personality_control", {})
    if personality.get("human_measurement") is not False:
        raise SystemExit("FairSynth synthetic OCEAN must never be labeled human measurement")
    reporting = fairsynth.get("reporting", {})
    if reporting.get("report_separately_from_real_world_datasets") is not True:
        raise SystemExit("FairSynth must remain separately reported")
    if reporting.get("no_meta_pooling_with_rq1_observed_demographics") is not True:
        raise SystemExit("FairSynth must never enter RQ1 observed-demographic meta-analysis")
    if reporting.get("no_meta_pooling_with_rq2_measured_psychometrics") is not True:
        raise SystemExit("FairSynth must never enter RQ2 measured-psychometric meta-analysis")

    extensions = experiment.get("extension_tracks", {})
    if set(extensions) != {"local_open_weight_transparency", "fairsynth360"}:
        raise SystemExit("experiment extension_tracks must contain local and FairSynth tracks only")
    if extensions["fairsynth360"].get("never_pool_with_real_demographic_or_psychometric_inference") is not True:
        raise SystemExit("experiment must prohibit pooling FairSynth with real-world inference")
    if set(extensions["local_open_weight_transparency"].get("families", [])) != set(local_by_family):
        raise SystemExit("experiment local families disagree with local_models.yaml")

    if set(study["rq3_reliability"].get("local_open_weight_families", [])) != set(local_by_family):
        raise SystemExit("study RQ3 local families disagree with local_models.yaml")
    if study.get("fairsynth360_scope", {}).get("users") != 360:
        raise SystemExit("study_design FairSynth scope must use all 360 synthetic users")
    if study["fairsynth360_scope"].get("never_pool_with_observed_demographic_estimates") is not True:
        raise SystemExit("study_design must prohibit FairSynth/observed-demographic pooling")

    rq4_selection = study["rq4_mitigation"].get("operating_point_selection", {})
    if rq4_selection.get("minimum_validation_utility_retention") != 0.95:
        raise SystemExit("RQ4 PAIR validation utility-retention floor must remain 0.95")
    if rq4_selection.get("global_single_operating_point") is not True:
        raise SystemExit("RQ4 must freeze one global PAIR operating point")
    if rq4_selection.get("per_model_or_dataset_tuning") is not False:
        raise SystemExit("RQ4 must prohibit per-model/per-dataset test tuning")
    if rq4_selection.get("selection_uses_test_outcomes") is not False:
        raise SystemExit("RQ4 PAIR selection must not use test outcomes")

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

    print("config preflight: six core dataset manifests/acquisition contracts agree")
    print("config preflight: RQ1/RQ4/MIND scopes match the executable core plan")
    print("config preflight: primary prompt/cue/order IDs agree across manifests")
    print("config preflight: six hosted provider semantics remain frozen")
    print("config preflight: two local model IDs/licenses/white-box semantics agree")
    print("config preflight: FairSynth-360 size/balance/separation guards agree")
    print("config preflight: contextual PAIR validation-only selection contract agrees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
