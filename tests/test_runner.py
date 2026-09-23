import json

import pytest

from faireval.providers.base import GenerationRequest, GenerationResponse, ProviderAdapter
from faireval.runner import run_one
from faireval.schema import Item, PromptCondition, UserInstance


class FakeProvider(ProviderAdapter):
    family = "fake"
    provider_name = "fake-provider"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def supports_seed(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        return GenerationResponse(
            text=next(self.outputs),
            requested_model_id=request.model_id,
            resolved_model_version="fake-v1",
            provider_metadata={
                "unit_test": True,
                "reasoning_or_thinking_applied": request.reasoning_or_thinking_setting,
                "sampling_controls_applied": True,
                "sampling_policy": "test",
                "output_token_parameter": "max_tokens",
            },
        )


def _instance():
    return UserInstance(
        dataset="toy",
        user_id="u1",
        history=[Item("h1", "Past")],
        candidates=[Item("i001", "A"), Item("i002", "B"), Item("i003", "C")],
        relevant_item_ids=frozenset({"i001"}),
    )


def _run(provider, tmp_path, **kwargs):
    return run_one(
        instance=_instance(),
        condition=PromptCondition("C0", "neutral"),
        provider=provider,
        model_id="fake",
        k=2,
        repetition=0,
        output_jsonl=tmp_path / "runs.jsonl",
        temperature=0.2,
        top_p=1.0,
        max_output_tokens=64,
        seed=7,
        reasoning_or_thinking_setting="disabled",
        **kwargs,
    )


def test_run_one_writes_v5_protocol_with_one_generation(tmp_path):
    provider = FakeProvider(['{"ranked_item_ids":["i001","i002"]}'])
    row = _run(provider, tmp_path)
    assert provider.calls == 1
    assert row["schema_version"] == "faireval-run-v5"
    assert row["strict_format_valid"] is True
    assert row["semantic_ranking_valid"] is True
    assert row["ranking"] == ["i001", "i002"]
    assert row["repair"] is None
    assert row["generative_format_repair_enabled"] is False
    assert row["provider_generation_calls_for_cell"] == 1
    disk = json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8").strip())
    assert disk["ranking"] == ["i001", "i002"]


def test_run_one_recovers_fenced_json_without_second_model_call(tmp_path):
    provider = FakeProvider([
        '```json\n{"ranked_item_ids":["i001","i002"]}\n```',
        '{"ranked_item_ids":["i003","i002"]}',
    ])
    row = _run(provider, tmp_path)
    assert provider.calls == 1
    assert row["strict_format_valid"] is False
    assert row["semantic_ranking_valid"] is True
    assert row["ranking"] == ["i001", "i002"]
    assert row["deterministic_normalization_applied"] is True
    assert row["normalization_actions"] == ["strip_single_recognized_json_fence"]
    assert "markdown_fence" in row["format_violations"]


def test_v5_rejects_generative_format_repair_before_provider_call(tmp_path):
    provider = FakeProvider(['{"ranked_item_ids":["i001","i002"]}'])
    with pytest.raises(ValueError, match="generative format repair is disabled"):
        _run(provider, tmp_path, allow_format_repair=True)
    assert provider.calls == 0
    assert not (tmp_path / "runs.jsonl").exists()
