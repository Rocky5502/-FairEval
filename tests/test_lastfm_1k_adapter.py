from faireval.datasets.lastfm_1k import LastFM1KAdapter


def _write_lastfm(root):
    (root / "userid-profile.tsv").write_text(
        "id\tgender\tage\tcountry\tregistered\n"
        "user_1\tf\t29\tNepal\tJan 1, 2007\n"
        "user_2\tm\t\tJapan\tJan 2, 2007\n",
        encoding="utf-8",
    )

    rows = []
    # Eligible user: six unique artists over six timestamps.
    for idx, artist in enumerate(["A", "B", "C", "D", "E", "F"], start=1):
        rows.append(
            "\t".join([
                "user_1",
                f"2009-01-0{idx}T00:00:00Z",
                f"mbid-{artist}",
                f"Artist {artist}",
                f"track-{artist}",
                f"Track {artist}",
            ])
        )
    # Ineligible profile still contributes global negative-candidate popularity.
    for idx, artist in enumerate(["G", "H", "I", "J"], start=1):
        rows.append(
            "\t".join([
                "user_2",
                f"2009-02-0{idx}T00:00:00Z",
                f"mbid-{artist}",
                f"Artist {artist}",
                f"track-{artist}",
                f"Track {artist}",
            ])
        )
    (root / "userid-timestamp-artid-artname-traid-traname.tsv").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_lastfm_adapter_builds_demographic_artist_task(tmp_path):
    _write_lastfm(tmp_path)
    adapter = LastFM1KAdapter(
        train_fraction=0.5,
        min_events=6,
        max_relevant_per_user=2,
        require_gender_and_age=True,
    )
    first = list(
        adapter.build_instances(
            tmp_path,
            users=1,
            candidate_set_size=4,
            max_history_items=3,
            seed=2027,
        )
    )
    second = list(
        adapter.build_instances(
            tmp_path,
            users=1,
            candidate_set_size=4,
            max_history_items=3,
            seed=2027,
        )
    )
    assert first == second
    assert len(first) == 1
    instance = first[0]
    assert instance.demographics == {
        "gender": "female",
        "age_group": "25_34",
        "country": "Nepal",
    }
    assert len(instance.history) == 3
    assert len(instance.candidates) == 4
    assert len(instance.relevant_item_ids) == 2
    assert set(instance.relevant_item_ids).issubset(set(instance.candidate_ids()))
    assert instance.instance_metadata["recommendation_unit"] == "artist"

    manifest = adapter.preprocessing_manifest()
    assert manifest["event_rows_seen"] == 10
    assert manifest["emitted_users"] == 1
    assert manifest["unique_artists"] == 10


def test_lastfm_card_marks_age_group_as_derived():
    card = LastFM1KAdapter().card()
    assert "age" in card.observed_fields
    assert "age_group" in card.derived_fields
    assert "country" in card.counterfactual_fields
