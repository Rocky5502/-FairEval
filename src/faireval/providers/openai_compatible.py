from __future__ import annotations

import os
from typing import Any

from .base import GenerationRequest, GenerationResponse, ProviderAdapter


class OpenAICompatibleAdapter(ProviderAdapter):
    """Adapter for providers exposing an OpenAI-compatible chat endpoint.

    Keep provider-specific model IDs/base URLs in config rather than hard-coding
    them in analysis code. The SDK import is lazy so metric-only workflows do
    not require provider packages.
    """

    def __init__(
        self,
        *,
        family: str,
        provider_name: str,
        api_key_env: str,
        base_url: str | None = None,
        supports_seed_flag: bool = False,
        json_mode: bool = True,
    ) -> None:
        self.family = family
        self.provider_name = provider_name
        self.api_key_env = api_key_env
        self.base_url = base_url
        self._supports_seed = supports_seed_flag
        self.json_mode = json_mode

    def supports_seed(self) -> bool:
        return self._supports_seed

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install the 'providers' optional dependency") from exc

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {self.api_key_env}")

        kwargs: dict[str, Any] = {
            "model": request.model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_output_tokens,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if request.seed is not None and self.supports_seed():
            kwargs["seed"] = request.seed

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        resolved = getattr(response, "model", None)
        usage = getattr(response, "usage", None)
        metadata = {
            "id": getattr(response, "id", None),
            "created": getattr(response, "created", None),
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
        }
        return GenerationResponse(
            text=text,
            requested_model_id=request.model_id,
            resolved_model_version=resolved,
            provider_metadata=metadata,
        )
