from __future__ import annotations

import os

from .anthropic_adapter import AnthropicAdapter
from .google_adapter import GoogleGenAIAdapter
from .local_transformers import LocalTransformersAdapter
from .openai_compatible import OpenAICompatibleAdapter


QWEN25_LOCAL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PHI35_LOCAL_REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"


def _gateway_enabled() -> bool:
    return os.environ.get("FAIREVAL_HOSTED_GATEWAY", "").strip().lower() == "zhizengzeng"


def _zhizengzeng_provider(family: str) -> OpenAICompatibleAdapter:
    """Route hosted FairEval families through Zhizengzeng's compatible API.

    One gateway key fixes endpoint provenance, while the immutable run plan keeps
    model-family/model-ID identity. The common gateway interface does not prove
    vendor-native deliberation or decoding semantics, so ``controls_verified``
    is false and the persisted run log records requested controls as unverified
    rather than pretending provider-native equivalence.
    """
    base_url = os.environ.get("ZZZ_BASE_URL", "https://api.zhizengzeng.com/v1")
    output_field = "max_completion_tokens" if family == "openai" else "max_tokens"
    return OpenAICompatibleAdapter(
        family=family,
        provider_name="zhizengzeng",
        api_key_env="ZZZ_API_KEY",
        base_url=base_url,
        supports_seed_flag=False,
        json_mode=True,
        output_token_parameter=output_field,
        controls_verified=False,
    )


def build_provider(family: str):
    """Build a provider adapter from frozen FairEval provider policy."""
    family = family.lower()

    if family in {"openai", "anthropic", "google", "deepseek", "qwen", "meta"} and _gateway_enabled():
        return _zhizengzeng_provider(family)

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
    if family == "qwen25_local":
        return LocalTransformersAdapter(
            family="qwen25_local",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            revision=QWEN25_LOCAL_REVISION,
            trust_remote_code=False,
            dtype_preference=os.environ.get("FAIREVAL_LOCAL_DTYPE", "bfloat16"),
        )
    if family == "phi35_local":
        return LocalTransformersAdapter(
            family="phi35_local",
            model_id="microsoft/Phi-3.5-mini-instruct",
            revision=PHI35_LOCAL_REVISION,
            trust_remote_code=True,
            dtype_preference=os.environ.get("FAIREVAL_LOCAL_DTYPE", "bfloat16"),
        )
    raise ValueError(f"Unsupported model family: {family}")
