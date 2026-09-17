from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .base import GenerationRequest, GenerationResponse, ProviderAdapter


class OpenAICompatibleAdapter(ProviderAdapter):
    """Adapter for providers exposing an OpenAI-compatible chat endpoint.

    ``output_token_parameter`` is the frozen study/provider contract recorded in
    the immutable plan. ``wire_output_token_parameter`` is the actual field sent
    over this compatible interface. They are normally identical, but a gateway
    can normalize a provider-native field (for example Gemini
    ``max_output_tokens``) onto the OpenAI-compatible ``max_tokens`` wire field.

    ``controls_verified`` is deliberately explicit: a gateway may accept an
    OpenAI-shaped request without proving that underlying vendor-native decoding
    controls were applied. FairEval records that uncertainty instead of treating
    request acceptance as provider-native equivalence.
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
        wire_output_token_parameter: str | None = None,
        controls_verified: bool = True,
    ) -> None:
        if output_token_parameter not in {
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
        }:
            raise ValueError("unsupported frozen output_token_parameter")
        wire = wire_output_token_parameter or output_token_parameter
        if wire not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                "wire_output_token_parameter must be 'max_tokens' or 'max_completion_tokens'"
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
        self.wire_output_token_parameter = wire
        self.controls_verified = bool(controls_verified)

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
        kwargs[self.wire_output_token_parameter] = request.max_output_tokens
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
            "sampling_controls_applied": True if self.controls_verified else None,
            "sampling_controls_verification": (
                "provider_interface_verified"
                if self.controls_verified
                else "gateway_request_accepted_underlying_native_application_unverified"
            ),
            "sampling_policy": (
                "explicit_temperature_and_top_p"
                if self.controls_verified
                else "gateway_explicit_temperature_top_p_requested_unverified_native_application"
            ),
            "output_token_parameter": self.output_token_parameter,
            "wire_output_token_parameter": self.wire_output_token_parameter,
            "provider_extra_body": dict(self.extra_body),
            "provider_extra_request_fields": dict(self.extra_request_fields),
            "reasoning_or_thinking_requested": request.reasoning_or_thinking_setting,
            "reasoning_or_thinking_applied": (
                request.reasoning_or_thinking_setting if self.controls_verified else None
            ),
            "reasoning_or_thinking_verification": (
                "provider_interface_verified"
                if self.controls_verified
                else "gateway_native_deliberation_control_unverified"
            ),
        }
        return GenerationResponse(
            text=text,
            requested_model_id=request.model_id,
            resolved_model_version=resolved,
            provider_metadata=metadata,
        )
