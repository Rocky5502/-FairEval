from __future__ import annotations

import os

from .anthropic_adapter import AnthropicAdapter
from .google_adapter import GoogleGenAIAdapter
from .openai_compatible import OpenAICompatibleAdapter


def build_provider(family: str):
    """Build a provider adapter from frozen FairEval provider policy.

    The core study uses the lowest practical deliberation mode for a direct
    ranking task: OpenAI reasoning ``none``; Claude thinking disabled; Gemini
    thinking ``low`` (its lowest supported 3.8-Flash level); DeepSeek thinking
    disabled; Qwen thinking disabled; and the base Llama checkpoint without an
    additional hosted reasoning wrapper. Any change requires a new study config.
    """
    family = family.lower()
    if family == "openai":
        return OpenAICompatibleAdapter(
            family="openai",
            provider_name="openai",
            api_key_env="OPENAI_API_KEY",
            supports_seed_flag=False,
            json_mode=True,
            extra_request_fields={"reasoning_effort": "none"},
            output_token_parameter="max_completion_tokens",
        )
    if family == "anthropic":
        return AnthropicAdapter()
    if family == "google":
        return GoogleGenAIAdapter()
    if family == "deepseek":
        return OpenAICompatibleAdapter(
            family="deepseek",
            provider_name="deepseek",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            supports_seed_flag=False,
            json_mode=True,
            extra_body={"thinking": {"type": "disabled"}},
            output_token_parameter="max_tokens",
        )
    if family == "qwen":
        base_url = os.environ.get("QWEN_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "Set QWEN_BASE_URL to the Model Studio endpoint matching the region/API key"
            )
        return OpenAICompatibleAdapter(
            family="qwen",
            provider_name="alibaba_model_studio",
            api_key_env="DASHSCOPE_API_KEY",
            base_url=base_url,
            supports_seed_flag=False,
            json_mode=True,
            extra_body={"enable_thinking": False},
            output_token_parameter="max_tokens",
        )
    if family == "meta":
        base_url = os.environ.get("LLAMA_BASE_URL")
        if not base_url:
            raise RuntimeError("Set LLAMA_BASE_URL for the frozen Llama hosting provider")
        return OpenAICompatibleAdapter(
            family="meta",
            provider_name=os.environ.get("LLAMA_PROVIDER_NAME", "configured_llama_host"),
            api_key_env="LLAMA_PROVIDER_API_KEY",
            base_url=base_url,
            supports_seed_flag=False,
            json_mode=True,
            output_token_parameter="max_tokens",
        )
    raise ValueError(f"Unsupported model family: {family}")
