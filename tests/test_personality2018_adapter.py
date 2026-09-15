import csv

import pytest

from faireval.datasets.personality2018 import Personality2018Adapter, score_personality2018


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_personality2018_emotional_stability_maps_to_neuroticism():
    row = {
        "openness": "7",
        "agreeableness": "4",
        "emotional_stability": "7",
        "conscientiousness": "1",
        "extraversion": "4",
    }
    profile = score_personality2018(row)
    assert profile.openness == pytest.approx(1.0)
    assert profile.agreeableness == pytest.approx(0.5)
    assert profile.neuroticism == pytest.approx(0.0)
    assert profile.conscientiousness == pytest.approx(0.0)
    assert profile.extraversion == pytest.approx(0.5)


def _write_dataset(root):
    _write_csv(
        root / "personality-data.csv",
        [
            "userid", "openness", "agreeableness", "emotional_stability",
            "conscientiousness", "extraversion",
        ],
        [{
            "userid": "hashed-user",
            "openness": "7",
            "agreeableness": "4",
            "emotional_stability": "7",
            "conscientiousness": "1",
            "extraversion": "4",
        }],
    )
    _write_csv(
        root / "movies.csv",
        ["movieId", "title", "genres"],
        [
            {"movieId": str(i), "title": f"Movie {i}", "genres": "Drama|Comedy"}
            for i in range(1, 11)
        ],
    )
    _write_csv(
        root / "ratings.csv",
        ["userid", "movie_id", "rating", "tstamp"],
        [
            {"userid": "hashed-user", "movie_id": str(i), "rating": str(r), "tstamp": str(i)}
            for i, r in [
                (1, 5), (2, 4), (3, 3), (4, 5), (5, 4), (6, 3), (7, 5), (8, 2)
            ]
        ],
    )


def test_personality2018_adapter_builds_grounded_task(tmp_path):
    _write_dataset(tmp_path)
    adapter = Personality2018Adapter(
        train_fraction=0.75,
        min_interactions=6,
        max_relevant_per_user=2,
    )
    first = list(
        adapter.build_instances(
            tmp_path,
            users=1,
            candidate_set_size=4,
            max_history_items=4,
            seed=2027,
        )
    )
    second = list(
        adapter.build_instances(
            tmp_path,
            users=1,
            candidate_set_size=4,
            max_history_items=4,
            seed=2027,
        )
    )
    assert first == second
    assert len(first) == 1
    instance = first[0]
    assert instance.personality is not None
    assert instance.personality.neuroticism == pytest.approx(0.0)
    assert "7" in instance.relevant_item_ids
    assert "8" in instance.candidate_ids()
    assert len(instance.candidates) == 4
    assert instance.instance_metadata["personality_raw_scale"] == "1_to_7"

    manifest = adapter.preprocessing_manifest()
    assert manifest["personality_profiles_loaded"] == 1
    assert manifest["emitted_users"] == 1
    assert "neuroticism=1-normalized emotional_stability" in manifest["personality_transformation"]
