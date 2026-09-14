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

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the 'providers' optional dependency") from exc

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {self.api_key_env}")

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=request.model_id,
            max_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            messages=[{"role": "user", "content": request.prompt}],
        )
        text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        text = "".join(text_parts)
        usage = getattr(response, "usage", None)
        return GenerationResponse(
            text=text,
            requested_model_id=request.model_id,
            resolved_model_version=getattr(response, "model", None),
            provider_metadata={
                "id": getattr(response, "id", None),
                "stop_reason": getattr(response, "stop_reason", None),
                "usage": {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                },
            },
        )
