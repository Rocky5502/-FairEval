from __future__ import annotations

from .base import DatasetAdapter
from .fairsynth360 import FairSynth360Adapter
from .lastfm_1k import LastFM1KAdapter
from .mind import MINDAdapter
from .movielens_1m import MovieLens1MAdapter
from .music_master_bfi2 import MusicMasterBFI2Adapter
from .personality2018 import Personality2018Adapter
from .reasoner import ReasonerAdapter


# Keep the original six real-world datasets as the confirmatory/core benchmark.
CORE_DATASET_IDS = (
    "personality2018",
    "music_master_bfi2",
    "reasoner",
    "movielens_1m",
    "lastfm_1k",
    "mind",
)
AUXILIARY_DATASET_IDS = ("fairsynth360",)
ALL_DATASET_IDS = CORE_DATASET_IDS + AUXILIARY_DATASET_IDS

# Backwards-compatible name used by the frozen six-dataset core planner.
DATASET_IDS = CORE_DATASET_IDS


def build_dataset_adapter(dataset_id: str) -> DatasetAdapter:
    """Return a versioned FairEval dataset adapter.

    The six real-world datasets remain the confirmatory/core benchmark.
    ``fairsynth360`` is a project-generated auxiliary stress-test dataset and is
    never silently pooled with observed-demographic or measured-psychometric
    claims from the real-world datasets.
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
    if key == "fairsynth360":
        return FairSynth360Adapter()
    raise ValueError(f"unknown dataset_id={dataset_id!r}; expected one of {ALL_DATASET_IDS!r}")
