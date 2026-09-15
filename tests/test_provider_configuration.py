import sys
import types
from types import SimpleNamespace

from faireval.providers.base import GenerationRequest
from faireval.providers.factory import build_provider
from faireval.providers.anthropic_adapter import AnthropicAdapter
from faireval.providers.google_adapter import GoogleGenAIAdapter


def _request(model_id: str, reasoning: str) -> GenerationRequest:
    return GenerationRequest(
        prompt="Return JSON only.",
        model_id=model_id,
        temperature=0.2,
        top_p=1.0,
        max_output_tokens=64,
        seed=None,
        reasoning_or_thinking_setting=reasoning,
    )


def test_openai_compatible_factory_freezes_reasoning_controls(monkeypatch):
    openai = build_provider("openai")
    assert openai.extra_request_fields == {"reasoning_effort": "none"}

    deepseek = build_provider("deepseek")
    assert deepseek.extra_body == {"thinking": {"type": "disabled"}}

    monkeypatch.setenv("QWEN_BASE_URL", "https://example.invalid/compatible-mode/v1")
    qwen = build_provider("qwen")
    assert qwen.extra_body == {"enable_thinking": False}

    monkeypatch.setenv("LLAMA_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLAMA_PROVIDER_NAME", "frozen_test_host")
    meta = build_provider("meta")
    assert meta.provider_name == "frozen_test_host"
    assert meta.base_url == "https://example.invalid/v1"


def test_claude_sonnet5_disables_thinking_and_omits_sampling(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"ranked_item_ids":["x"]}')],
            model="claude-sonnet-5",
            id="msg_test",
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = lambda api_key: SimpleNamespace(
        messages=SimpleNamespace(create=create)
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    response = AnthropicAdapter().generate(_request("claude-sonnet-5", "disabled"))
    assert captured["thinking"] == {"type": "disabled"}
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert response.provider_metadata["sampling_controls_applied"] is False
    assert response.provider_metadata["reasoning_or_thinking_applied"] == "disabled"


def test_gemini38_uses_low_thinking_and_no_deprecated_sampling(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output_text='{"ranked_item_ids":["x"]}',
            model="gemini-3.8-flash",
            id="interaction_test",
            status="completed",
            usage=None,
        )

    fake_google = types.ModuleType("google")
    fake_google.genai = SimpleNamespace(
        Client=lambda api_key: SimpleNamespace(interactions=SimpleNamespace(create=create))
    )
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    response = GoogleGenAIAdapter().generate(_request("gemini-3.8-flash", "low"))
    assert captured["generation_config"]["thinking_level"] == "low"
    assert "temperature" not in captured["generation_config"]
    assert "top_p" not in captured["generation_config"]
    assert response.provider_metadata["sampling_controls_applied"] is False
    assert response.provider_metadata["reasoning_or_thinking_applied"] == "low"
