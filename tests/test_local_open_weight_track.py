import inspect
from pathlib import Path

from faireval.datasets.fairsynth360 import FairSynth360Adapter
from faireval.local_plan import load_local_model_panel, plan_fairsynth_conditions
from faireval.providers.factory import build_provider
from faireval.providers.local_transformers import LocalTransformersAdapter


ROOT = Path(__file__).resolve().parents[1]


def _synth_instances(n: int = 12):
    return list(
        FairSynth360Adapter().build_instances(
            Path("."),
            users=n,
            candidate_set_size=30,
            max_history_items=8,
            seed=1729,
        )
    )


def test_local_model_manifest_is_exact_two_model_panel():
    panel = load_local_model_panel(ROOT / "configs" / "local_models.yaml")
    assert {row["family"] for row in panel} == {"qwen25_local", "phi35_local"}
    assert {row["model_id"] for row in panel} == {
        "Qwen/Qwen2.5-7B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
    }
    assert {row["output_token_parameter"] for row in panel} == {"max_new_tokens"}
    assert {row["quantization_policy"] for row in panel} == {"bitsandbytes_nf4_4bit"}
    assert {row["attn_implementation"] for row in panel} == {"eager"}
    assert {row["kv_cache_policy"] for row in panel} == {
        "enabled",
        "disabled_for_transformers_compatibility",
    }


def test_local_provider_factory_is_lazy_and_white_box_capable():
    qwen = build_provider("qwen25_local")
    phi = build_provider("phi35_local")
    assert qwen.provider_name == "local_transformers"
    assert phi.provider_name == "local_transformers"
    assert qwen.output_token_parameter == "max_new_tokens"
    assert phi.output_token_parameter == "max_new_tokens"
    assert qwen.supports_seed() is True
    assert phi.supports_seed() is True
    assert qwen.quantization_policy == "bitsandbytes_nf4_4bit"
    assert phi.quantization_policy == "bitsandbytes_nf4_4bit"
    assert qwen.attn_implementation == "eager"
    assert phi.attn_implementation == "eager"
    assert qwen.disable_kv_cache is False
    assert phi.disable_kv_cache is True
    # Construction must not load multi-GB weights in unit tests/CI.
    assert qwen._model is None
    assert phi._model is None


def test_fairsynth_plan_contains_identity_and_personality_controls():
    rows = plan_fairsynth_conditions(_synth_instances(), seed=1729)
    by_user = {}
    for row in rows:
        by_user.setdefault(row.user_id, []).append(row)
    assert len(by_user) == 12
    for user_rows in by_user.values():
        ids = [row.condition.condition_id for row in user_rows]
        assert "C0" in ids
        assert "C1" in ids
        assert "C3" in ids
        assert "C4" in ids
        assert sum(value.startswith("C2:") for value in ids) == 2
        assert all(row.confirmatory is False for row in user_rows)


def test_local_generate_avoids_unsupported_transformers_generator_kwarg():
    source = inspect.getsource(LocalTransformersAdapter.generate)
    assert 'kwargs["generator"]' not in source
    assert "torch.random.fork_rng" in source
    assert "torch.manual_seed" in source
