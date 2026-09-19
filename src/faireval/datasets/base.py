from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from ..schema import UserInstance


@dataclass(frozen=True)
class DatasetCard:
    dataset_id: str
    source: str
    version: str
    license: str
    domain: str
    track: str
    observed_fields: tuple[str, ...]
    derived_fields: tuple[str, ...] = ()
    counterfactual_fields: tuple[str, ...] = ()
    notes: str = ""


class DatasetAdapter(ABC):
    """Contract for turning a public recommendation dataset into fixed tasks."""

    dataset_id: str

    @abstractmethod
    def card(self) -> DatasetCard:
        raise NotImplementedError

    @abstractmethod
    def build_instances(
        self,
        raw_dir: Path,
        *,
        users: int,
        candidate_set_size: int,
        max_history_items: int,
        seed: int,
    ) -> Iterator[UserInstance]:
        """Yield deterministic, leakage-checked user instances."""
        raise NotImplementedError

    def preprocessing_manifest(self) -> Mapping[str, object]:
        """Dataset-specific adapters should override with filtering/split details."""
        return {"dataset_id": self.dataset_id}
