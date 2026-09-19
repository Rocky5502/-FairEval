from faireval.local_robustness_plan import freeze_local_robustness_seeds


def _cell(*, cue_id: str = "structured_key_value", family: str = "qwen25_local"):
    return {
        "schema_version": "faireval-run-plan-v1",
        "cell_id": "preseed-placeholder",
        "dataset": "movielens_1m",
        "user_id": "u1",
        "condition": {
            "condition_id": "C1",
            "condition_name": "observed_demographic",
            "demographics": {"gender": "F"},
            "personality": None,
            "intervention": {"kind": "observed_demographic"},
        },
        "analysis_roles": ["rq3_reliability", "rq3_cue"],
        "confirmatory": False,
        "model_family": family,
        "model_id": (
            "Qwen/Qwen2.5-7B-Instruct"
            if family == "qwen25_local"
            else "microsoft/Phi-3.5-mini-instruct"
        ),
        "reasoning_or_thinking_setting": "not_applicable",
        "sampling_policy": "explicit_temperature_and_top_p",
        "output_token_parameter": "max_new_tokens",
        "template_id": "field_v2_a",
        "prompt_mode": "audit",
        "cue_id": cue_id,
        "candidate_order_seed": None,
        "k": 10,
        "repetition": 0,
        "temperature": 0.2,
        "top_p": 1.0,
        "max_output_tokens": 512,
        "robustness_factor": "cue",
        "robustness_level": cue_id,
    }


def test_local_robustness_seed_freeze_is_deterministic_and_rehashes_cells():
    first = freeze_local_robustness_seeds([_cell()], experiment_seed=1729)
    second = freeze_local_robustness_seeds([_cell()], experiment_seed=1729)
    assert first == second
    assert isinstance(first[0]["seed"], int)
    assert 0 <= first[0]["seed"] < 2**32
    assert len(first[0]["cell_id"]) == 64
    assert first[0]["cell_id"] != "preseed-placeholder"


def test_different_robustness_levels_receive_different_local_seeds_and_ids():
    rows = freeze_local_robustness_seeds(
        [
            _cell(cue_id="first_person_disclosure"),
            _cell(cue_id="profile_sentence"),
        ],
        experiment_seed=1729,
    )
    assert rows[0]["seed"] != rows[1]["seed"]
    assert rows[0]["cell_id"] != rows[1]["cell_id"]


def test_local_robustness_rejects_hosted_or_wrong_token_contracts():
    hosted = _cell(family="qwen25_local")
    hosted["model_family"] = "qwen"
    try:
        freeze_local_robustness_seeds([hosted], experiment_seed=1729)
    except ValueError as exc:
        assert "unexpected local robustness model family" in str(exc)
    else:
        raise AssertionError("hosted family must be rejected")

    wrong = _cell()
    wrong["output_token_parameter"] = "max_tokens"
    try:
        freeze_local_robustness_seeds([wrong], experiment_seed=1729)
    except ValueError as exc:
        assert "max_new_tokens" in str(exc)
    else:
        raise AssertionError("wrong local output-token contract must be rejected")
