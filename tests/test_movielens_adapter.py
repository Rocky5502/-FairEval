from faireval.datasets.movielens_1m import MovieLens1MAdapter


def _write_ml1m(root):
    (root / "users.dat").write_text(
        "1::F::25::3::00000\n"
        "2::M::35::7::11111\n",
        encoding="latin-1",
    )
    (root / "movies.dat").write_text(
        "1::Movie 1::Drama\n"
        "2::Movie 2::Comedy\n"
        "3::Movie 3::Action\n"
        "4::Movie 4::Drama\n"
        "5::Movie 5::Comedy\n"
        "6::Movie 6::Action\n"
        "7::Movie 7::Drama\n"
        "8::Movie 8::Comedy\n"
        "9::Movie 9::Action\n"
        "10::Movie 10::Drama\n",
        encoding="latin-1",
    )
    # User 1 has 8 chronological interactions. With train_fraction=.75 the
    # last two are held out; item 7 is relevant and item 8 is an explicit hard negative.
    rows = [
        "1::1::5::1",
        "1::2::4::2",
        "1::3::3::3",
        "1::4::5::4",
        "1::5::4::5",
        "1::6::3::6",
        "1::7::5::7",
        "1::8::2::8",
        # User 2 contributes popularity counts and additional unseen catalog support.
        "2::1::4::1",
        "2::2::4::2",
        "2::3::4::3",
        "2::4::4::4",
        "2::9::5::5",
        "2::10::5::6",
    ]
    (root / "ratings.dat").write_text("\n".join(rows) + "\n", encoding="latin-1")


def test_movielens_adapter_builds_deterministic_leakage_checked_task(tmp_path):
    _write_ml1m(tmp_path)
    adapter = MovieLens1MAdapter(
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

    assert len(first) == 1
    assert first == second
    instance = first[0]
    assert instance.dataset == "movielens_1m"
    assert instance.demographics == {"gender": "female", "age_group": "25_34"}
    assert "7" in instance.relevant_item_ids
    assert set(instance.relevant_item_ids).issubset(set(instance.candidate_ids()))
    assert len(instance.candidates) == 4
    assert len(set(instance.candidate_ids())) == 4
    assert len(instance.history) == 4
    assert all("rating" in item.metadata for item in instance.history)
    assert "8" in instance.candidate_ids()  # held-out explicit dislike retained as a hard negative

    manifest = adapter.preprocessing_manifest()
    assert manifest["emitted_users"] == 1
    assert manifest["candidate_set_size"] == 4
    assert "zip_code" in manifest["fields_intentionally_excluded"]


def test_movielens_card_distinguishes_observed_and_counterfactual_fields():
    card = MovieLens1MAdapter().card()
    assert "gender" in card.observed_fields
    assert "age_group" in card.observed_fields
    assert set(card.counterfactual_fields) == {"gender", "age_group"}
    assert "ZIP" in card.notes
