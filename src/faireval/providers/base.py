from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    model_id: str
    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int | None = None
    reasoning_or_thinking_setting: str | None = None


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    requested_model_id: str
    resolved_model_version: str | None
    provider_metadata: Mapping[str, Any]


class ProviderAdapter(ABC):
    """Minimal interface that prevents provider details leaking into analysis code."""

    family: str
    provider_name: str

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError

    @abstractmethod
    def supports_seed(self) -> bool:
        raise NotImplementedError
