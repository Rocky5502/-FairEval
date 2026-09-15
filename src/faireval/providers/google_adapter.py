from __future__ import annotations

import os

from .base import GenerationRequest, GenerationResponse, ProviderAdapter


class GoogleGenAIAdapter(ProviderAdapter):
    family = "google"
    provider_name = "google_gemini_api"

    def __init__(self, api_key_env: str = "GEMINI_API_KEY") -> None:
        self.api_key_env = api_key_env

    def supports_seed(self) -> bool:
        # The Interactions API exposes a seed, but FairEval does not assume
        # cross-provider seed equivalence. Repeated generations are the primary
        # stochasticity estimator in the six-family analysis.
        return False

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the 'providers' optional dependency") from exc

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {self.api_key_env}")

        client = genai.Client(api_key=api_key)
        # Gemini 3.8 Flash deprecates temperature/top_p/top_k and does not support
        # fully disabling thinking. FairEval therefore freezes the lowest
        # supported thinking level ('low') and records that sampling controls were
        # requested by the cross-provider protocol but not applied by Gemini.
        interaction = client.interactions.create(
            model=request.model_id,
            input=request.prompt,
            generation_config={
                "thinking_level": "low",
                "max_output_tokens": request.max_output_tokens,
            },
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": {
                    "type": "object",
                    "properties": {
                        "ranked_item_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["ranked_item_ids"],
                    "additionalProperties": False,
                },
            },
        )
        usage = getattr(interaction, "usage", None)
        usage_payload = None
        if usage is not None:
            if hasattr(usage, "model_dump"):
                usage_payload = usage.model_dump()
            else:
                usage_payload = {
                    "total_input_tokens": getattr(usage, "total_input_tokens", None),
                    "total_output_tokens": getattr(usage, "total_output_tokens", None),
                    "total_thought_tokens": getattr(usage, "total_thought_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }

        return GenerationResponse(
            text=getattr(interaction, "output_text", "") or "",
            requested_model_id=request.model_id,
            resolved_model_version=getattr(interaction, "model", None),
            provider_metadata={
                "interaction_id": getattr(interaction, "id", None),
                "status": getattr(interaction, "status", None),
                "reasoning_or_thinking_applied": "low",
                "sampling_controls_requested": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                },
                "sampling_controls_applied": False,
                "sampling_policy": "provider_default_sampling_controls_deprecated",
                "output_token_parameter": "max_output_tokens",
                "usage": usage_payload,
            },
        )
