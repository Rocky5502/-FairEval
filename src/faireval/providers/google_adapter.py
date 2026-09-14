from __future__ import annotations

import os

from .base import GenerationRequest, GenerationResponse, ProviderAdapter


class GoogleGenAIAdapter(ProviderAdapter):
    family = "google"
    provider_name = "google_gemini_api"

    def __init__(self, api_key_env: str = "GEMINI_API_KEY") -> None:
        self.api_key_env = api_key_env

    def supports_seed(self) -> bool:
        # Treat reproducibility through repeated sampling; do not assume provider
        # seed semantics unless explicitly verified for the frozen API version.
        return False

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the 'providers' optional dependency") from exc

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {self.api_key_env}")

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_output_tokens,
            response_mime_type="application/json",
        )
        response = client.models.generate_content(
            model=request.model_id,
            contents=request.prompt,
            config=config,
        )
        return GenerationResponse(
            text=response.text or "",
            requested_model_id=request.model_id,
            resolved_model_version=getattr(response, "model_version", None),
            provider_metadata={
                "response_id": getattr(response, "response_id", None),
                "usage_metadata": (
                    response.usage_metadata.model_dump()
                    if getattr(response, "usage_metadata", None) is not None
                    and hasattr(response.usage_metadata, "model_dump")
                    else None
                ),
            },
        )
