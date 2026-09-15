from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .base import GenerationRequest, GenerationResponse, ProviderAdapter


class OpenAICompatibleAdapter(ProviderAdapter):
    """Adapter for providers exposing an OpenAI-compatible chat endpoint.

    Provider-specific non-standard controls (for example DeepSeek ``thinking``
    or Qwen ``enable_thinking``) are supplied at construction time as a frozen
    ``extra_body`` mapping. The output-token field is also explicit because the
    current OpenAI API uses ``max_completion_tokens`` while several compatible
    providers still document ``max_tokens``.
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
        extra_body: Mapping[str, Any] | None = None,
        extra_request_fields: Mapping[str, Any] | None = None,
        output_token_parameter: str = "max_tokens",
    ) -> None:
        if output_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                "output_token_parameter must be 'max_tokens' or 'max_completion_tokens'"
            )
        self.family = family
        self.provider_name = provider_name
        self.api_key_env = api_key_env
        self.base_url = base_url
        self._supports_seed = supports_seed_flag
        self.json_mode = json_mode
        self.extra_body = dict(extra_body or {})
        self.extra_request_fields = dict(extra_request_fields or {})
        self.output_token_parameter = output_token_parameter

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
            **self.extra_request_fields,
        }
        kwargs[self.output_token_parameter] = request.max_output_tokens
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if request.seed is not None and self.supports_seed():
            kwargs["seed"] = request.seed
        if self.extra_body:
            kwargs["extra_body"] = dict(self.extra_body)

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
            "sampling_controls_requested": {
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
            "sampling_controls_applied": True,
            "sampling_policy": "explicit_temperature_and_top_p",
            "output_token_parameter": self.output_token_parameter,
            "provider_extra_body": dict(self.extra_body),
            "provider_extra_request_fields": dict(self.extra_request_fields),
            "reasoning_or_thinking_applied": request.reasoning_or_thinking_setting,
        }
        return GenerationResponse(
            text=text,
            requested_model_id=request.model_id,
            resolved_model_version=resolved,
            provider_metadata=metadata,
        )
