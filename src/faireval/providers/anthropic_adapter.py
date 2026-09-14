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
    def _uses_provider_default_sampling(model_id: str) -> bool:
        """Whether FairEval must omit deprecated sampling controls.

        Anthropic documents ``temperature``, ``top_p``, and ``top_k`` as
        deprecated for Claude Opus 4.7 and later. The current FairEval panel uses
        Claude Sonnet 5, for which we therefore omit explicit sampling controls
        and record that provider defaults were applied. This is more honest than
        pretending all six providers expose equivalent decoding knobs.
        """
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
        use_defaults = self._uses_provider_default_sampling(request.model_id)
        create_kwargs = {
            "model": request.model_id,
            "max_tokens": request.max_output_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if not use_defaults:
            # Retained for older compatible Claude models used only in explicitly
            # versioned auxiliary experiments. The ECIR core Sonnet-5 run takes
            # the provider-default branch above.
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
                "sampling_controls_requested": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                },
                "sampling_controls_applied": not use_defaults,
                "sampling_policy": (
                    "provider_default_deprecated_controls_omitted"
                    if use_defaults
                    else "explicit_temperature_and_top_p"
                ),
                "usage": {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                },
            },
        )
