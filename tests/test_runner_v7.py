import json

from faireval.providers.base import GenerationRequest, GenerationResponse, ProviderAdapter
from faireval.runner import run_one
from faireval.schema import Item, PromptCondition, UserInstance


class Provider(ProviderAdapter):
    family = "fake"
    provider_name = "fake"

    def supports_seed(self):
        return True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text='{"ranked_item_ids":["C02","C01"]}',
            requested_model_id=request.model_id,
            resolved_model_version="fake-v1",
            provider_metadata={
                "reasoning_or_thinking_applied": request.reasoning_or_thinking_setting,
                "sampling_controls_applied": True,
                "sampling_policy": "test",
                "output_token_parameter": "max_tokens",
            },
        )


def test_v7_handle_decoding(tmp_path):
    instance = UserInstance(
        dataset="toy",
        user_id="u1",
        history=[Item("h1", "History")],
        candidates=[Item("i001", "A"), Item("i002", "B"), Item("i003", "C")],
        relevant_item_ids=frozenset({"i001"}),
    )
    row = run_one(
        instance=instance,
        condition=PromptCondition("C0", "neutral"),
        provider=Provider(),
        model_id="fake",
        k=2,
        repetition=0,
        output_jsonl=tmp_path / "runs.jsonl",
        temperature=0.2,
        top_p=1.0,
        max_output_tokens=64,
        seed=7,
        reasoning_or_thinking_setting="disabled",
        run_schema_version="faireval-run-v7",
        prompt_interface_version="faireval-prompt-interface-v7",
    )
    assert row["semantic_ranking_valid"] is True
    assert row["ranking_selection_ids"] == ["C02", "C01"]
    assert row["ranking"] == ["i002", "i001"]
    disk = json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8"))
    assert disk["ranking"] == ["i002", "i001"]
