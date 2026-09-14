from __future__ import annotations

from .base import DatasetAdapter
from .lastfm_1k import LastFM1KAdapter
from .mind import MINDAdapter
from .movielens_1m import MovieLens1MAdapter
from .music_master_bfi2 import MusicMasterBFI2Adapter
from .personality2018 import Personality2018Adapter
from .reasoner import ReasonerAdapter


DATASET_IDS = (
    "personality2018",
    "music_master_bfi2",
    "reasoner",
    "movielens_1m",
    "lastfm_1k",
    "mind",
)


def build_dataset_adapter(dataset_id: str) -> DatasetAdapter:
    """Return the frozen FairEval adapter for one benchmark dataset.

    Dataset-specific non-default thresholds must be changed explicitly in code or
    a versioned config before a run; this factory deliberately does not accept an
    unstructured **kwargs bag that could create undocumented experiments.
    """
    key = dataset_id.strip().lower()
    if key == "personality2018":
        return Personality2018Adapter()
    if key == "music_master_bfi2":
        return MusicMasterBFI2Adapter()
    if key == "reasoner":
        return ReasonerAdapter()
    if key == "movielens_1m":
        return MovieLens1MAdapter()
    if key == "lastfm_1k":
        return LastFM1KAdapter()
    if key == "mind":
        return MINDAdapter()
    raise ValueError(f"unknown dataset_id={dataset_id!r}; expected one of {DATASET_IDS!r}")
