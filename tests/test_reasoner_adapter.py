import csv

import pytest

from faireval.datasets.reasoner import ReasonerAdapter, score_cbf_pi_15


def test_cbf_pi_15_reverse_scoring_is_exact():
    row = {f"Q{i}": "5" for i in range(1, 16)}
    profile = score_cbf_pi_15(row)
    # All non-extraversion dimensions use only non-reversed items here.
    assert profile.openness == pytest.approx(1.0)
    assert profile.conscientiousness == pytest.approx(1.0)
    assert profile.agreeableness == pytest.approx(1.0)
    assert profile.neuroticism == pytest.approx(1.0)
    # Q2 and Q5 become zero after reverse scoring; Q14 remains five.
    assert profile.extraversion == pytest.approx(1.0 / 3.0)


def _write_tsv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_reasoner(root):
    _write_tsv(
        root / "user.csv",
        ["user_id", "age", "gender", "education", "career", "income", "address", "hobby"],
        [{
            "user_id": "1",
            "age": "3",
            "gender": "0",
            "education": "2",
            "career": "4",
            "income": "1",
            "address": "9",
            "hobby": "music",
        }],
    )

    _write_tsv(
        root / "video.csv",
        ["video_id", "title", "info", "tags", "duration", "category"],
        [
            {
                "video_id": str(i),
                "title": f"Video {i}",
                "info": f"Info {i}",
                "tags": f"[{i}]",
                "duration": "60",
                "category": str(i % 3),
            }
            for i in range(1, 11)
        ],
    )

    interaction_fields = [
        "user_id", "video_id", "like", "persuasiveness_tag", "rating", "review",
        "informativeness_tag", "satisfaction_tag", "watch_again",
    ]
    _write_tsv(
        root / "interaction.csv",
        interaction_fields,
        [
            {
                "user_id": "1",
                "video_id": str(i),
                "like": "1",
                "persuasiveness_tag": "[]",
                "rating": "5.0",
                "review": "good",
                "informativeness_tag": "[]",
                "satisfaction_tag": "[]",
                "watch_again": "0",
            }
            for i in range(1, 7)
        ],
    )

    bigfive_fields = ["user_id"] + [f"Q{i}" for i in range(1, 16)]
    bigfive_row = {"user_id": "1", **{f"Q{i}": "5" for i in range(1, 16)}}
    _write_tsv(root / "bigfive.csv", bigfive_fields, [bigfive_row])


def test_reasoner_adapter_builds_measured_personality_task(tmp_path):
    _write_reasoner(tmp_path)
    adapter = ReasonerAdapter(
        train_fraction=0.5,
        min_interactions=4,
        max_relevant_per_user=2,
    )
    instances = list(
        adapter.build_instances(
            tmp_path,
            users=1,
            candidate_set_size=4,
            max_history_items=3,
            seed=2027,
        )
    )
    assert len(instances) == 1
    instance = instances[0]
    assert instance.personality is not None
    assert instance.personality.extraversion == pytest.approx(1.0 / 3.0)
    assert instance.demographics["gender"] == "female"
    assert len(instance.candidates) == 4
    assert len(instance.relevant_item_ids) == 2
    assert set(instance.relevant_item_ids).issubset(set(instance.candidate_ids()))
    assert instance.instance_metadata["personality_instrument"] == "CBF-PI-15"

    manifest = adapter.preprocessing_manifest()
    assert manifest["personality_instrument"] == "CBF-PI-15"
    assert manifest["reverse_scored_items"] == [2, 5]
    assert manifest["emitted_users"] == 1
