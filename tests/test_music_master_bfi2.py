from pathlib import Path

import pandas as pd
import pytest

from faireval.datasets.music_master_bfi2 import MusicMasterBFI2Adapter


def _write_fixture(path: Path) -> None:
    rows = []
    # Two users have overlapping but non-identical song sets, mimicking the
    # sparse public matrix and guaranteeing a pool of unseen candidate negatives.
    profiles = {
        "u1": (4.0, 3.0, 2.0, 4.5, 2.5),
        "u2": (2.0, 4.0, 4.5, 3.0, 3.5),
    }
    song_indices = {
        "u1": range(0, 12),
        "u2": range(6, 18),
    }
    for user_id, profile in profiles.items():
        for index in song_indices[user_id]:
            rows.append(
                {
                    "user_id": user_id,
                    "song_id": f"s{index}",
                    "title": f"Song {index}",
                    "genre": "jazz" if index % 2 == 0 else "rock",
                    # Keep every held-out row relevant so the deterministic hash
                    # split cannot accidentally produce a relevance-free test set.
                    "Q1": 5,
                    "Q2": 4 if index % 2 == 0 else 2,
                    "Q3": 3 if index % 2 == 0 else 1,
                    "Openness": profile[0],
                    "Conscientiousness": profile[1],
                    "Extraversion": profile[2],
                    "Agreeableness": profile[3],
                    "Neuroticism": profile[4],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_music_master_builds_deterministic_personality_instances(tmp_path: Path):
    source = tmp_path / "music_master.csv"
    _write_fixture(source)
    adapter = MusicMasterBFI2Adapter(
        min_interactions=10,
        train_fraction=0.75,
        max_relevant_per_user=2,
    )

    first = list(
        adapter.build_instances(
            tmp_path,
            users=2,
            candidate_set_size=6,
            max_history_items=5,
            seed=1729,
        )
    )
    second = list(
        adapter.build_instances(
            tmp_path,
            users=2,
            candidate_set_size=6,
            max_history_items=5,
            seed=1729,
        )
    )

    assert [instance.user_id for instance in first] == [instance.user_id for instance in second]
    assert [instance.candidate_ids() for instance in first] == [
        instance.candidate_ids() for instance in second
    ]
    assert len(first) == 2
    for instance in first:
        assert instance.personality is not None
        values = instance.personality.as_dict()
        assert all(0.0 <= value <= 1.0 for value in values.values())
        assert len(instance.candidates) == 6
        assert instance.relevant_item_ids
        assert instance.instance_metadata["personality_instrument"] == "BFI-2"
        assert instance.instance_metadata["primary_rating"] == "Q1"

    manifest = adapter.preprocessing_manifest()
    assert manifest["primary_rating"] == "Q1"
    assert manifest["users_loaded"] == 2
    assert manifest["items_loaded"] == 18
    assert manifest["semantic_item_fields_available"] == ["genre", "title"]


def test_music_master_rejects_ambiguous_or_missing_personality_schema(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "user_id": ["u1"],
            "song_id": ["s1"],
            "Q1": [5],
            "Openness": [4],
            "Conscientiousness": [3],
            "Extraversion": [2],
            "Agreeableness": [4],
            # Neuroticism intentionally missing.
        }
    )
    frame.to_csv(tmp_path / "music_master.csv", index=False)
    adapter = MusicMasterBFI2Adapter(min_interactions=1)
    with pytest.raises(ValueError, match="neuroticism"):
        list(
            adapter.build_instances(
                tmp_path,
                users=1,
                candidate_set_size=2,
                max_history_items=1,
                seed=1,
            )
        )


def test_music_master_rejects_profile_values_outside_declared_scale(tmp_path: Path):
    source = tmp_path / "music_master.csv"
    _write_fixture(source)
    frame = pd.read_csv(source)
    frame.loc[0, "Openness"] = 9.0
    frame.to_csv(source, index=False)

    adapter = MusicMasterBFI2Adapter(min_interactions=1)
    with pytest.raises(ValueError, match="outside declared BFI-2 scale"):
        list(
            adapter.build_instances(
                tmp_path,
                users=1,
                candidate_set_size=2,
                max_history_items=1,
                seed=1,
            )
        )
