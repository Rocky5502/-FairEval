from __future__ import annotations

import os

from .base import GenerationRequest, GenerationResponse, ProviderAdapter


class AnthropicAdapter(ProviderAdapter):
    family = "anthropic"
    provider_name = "anthropic"

    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY") -> None:
        self.api_key_env = api_key_env

    def supports_seed(self) -> bool:
        return False

    @staticmethod
    def _is_current_claude5(model_id: str) -> bool:
        normalized = model_id.strip().lower()
        return normalized.startswith("claude-sonnet-5") or normalized.startswith("claude-opus-5")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the 'providers' optional dependency") from exc

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {self.api_key_env}")

        client = anthropic.Anthropic(api_key=api_key)
        is_claude5 = self._is_current_claude5(request.model_id)
        create_kwargs = {
            "model": request.model_id,
            "max_tokens": request.max_output_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if is_claude5:
            # Claude Sonnet 5 defaults to adaptive thinking. FairEval explicitly
            # disables it so the benchmark measures direct ranking behavior. The
            # same generation also rejects non-default temperature/top_p/top_k,
            # so those controls are deliberately omitted rather than faked.
            create_kwargs["thinking"] = {"type": "disabled"}
        else:
            # Auxiliary older-Claude experiments may retain explicit controls,
            # but they are not part of the frozen ECIR six-family core panel.
            create_kwargs["temperature"] = request.temperature
            create_kwargs["top_p"] = request.top_p

        response = client.messages.create(**create_kwargs)
        text_parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        text = "".join(text_parts)
        usage = getattr(response, "usage", None)
        return GenerationResponse(
            text=text,
            requested_model_id=request.model_id,
            resolved_model_version=getattr(response, "model", None),
            provider_metadata={
                "id": getattr(response, "id", None),
                "stop_reason": getattr(response, "stop_reason", None),
                "reasoning_or_thinking_applied": "disabled" if is_claude5 else None,
                "sampling_controls_requested": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                },
                "sampling_controls_applied": not is_claude5,
                "sampling_policy": (
                    "provider_default_sampling_nondefault_controls_deprecated"
                    if is_claude5
                    else "explicit_temperature_and_top_p"
                ),
                "usage": {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                },
            },
        )
