import json

from faireval.providers.base import GenerationRequest, GenerationResponse, ProviderAdapter
from faireval.runner import run_one
from faireval.schema import Item, PromptCondition, UserInstance


class FakeProvider(ProviderAdapter):
    family = "fake"
    provider_name = "fake-provider"

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def supports_seed(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text=next(self.outputs),
            requested_model_id=request.model_id,
            resolved_model_version="fake-v1",
            provider_metadata={"unit_test": True},
        )


def _instance():
    return UserInstance(
        dataset="toy",
        user_id="u1",
        history=[Item("h1", "Past")],
        candidates=[Item("a", "A"), Item("b", "B"), Item("c", "C")],
        relevant_item_ids=frozenset({"a"}),
    )


def test_run_one_writes_valid_jsonl(tmp_path):
    provider = FakeProvider(['{"ranked_item_ids":["a","b"]}'])
    path = tmp_path / "runs.jsonl"
    row = run_one(
        instance=_instance(),
        condition=PromptCondition("C0", "neutral"),
        provider=provider,
        model_id="fake",
        k=2,
        repetition=0,
        output_jsonl=path,
        temperature=0.2,
        top_p=1.0,
        max_output_tokens=64,
        seed=7,
        allow_format_repair=False,
    )
    assert row["final_valid"]
    disk = json.loads(path.read_text(encoding="utf-8").strip())
    assert disk["ranking"] == ["a", "b"]
    assert disk["seed_supported"] is True
    assert len(disk["prompt_sha256"]) == 64


def test_run_one_retains_invalid_first_response_and_repair(tmp_path):
    provider = FakeProvider([
        "a, b",
        '{"ranked_item_ids":["a","b"]}',
    ])
    row = run_one(
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
        allow_format_repair=True,
    )
    assert not row["initial_valid"]
    assert row["raw_response"] == "a, b"
    assert row["repair"] is not None
    assert row["repair"]["valid"]
    assert row["final_valid"]
