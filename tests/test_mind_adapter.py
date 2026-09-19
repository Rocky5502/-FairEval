from faireval.datasets.mind import MINDAdapter


def _write_mind(root):
    (root / "news.tsv").write_text(
        "N1\tnews\tworld\tTitle 1\tAbstract 1\t[]\t[]\n"
        "N2\tnews\tworld\tTitle 2\tAbstract 2\t[]\t[]\n"
        "N3\tsports\tfootball\tTitle 3\tAbstract 3\t[]\t[]\n"
        "N4\tsports\tfootball\tTitle 4\tAbstract 4\t[]\t[]\n"
        "N5\tfinance\tmarkets\tTitle 5\tAbstract 5\t[]\t[]\n"
        "N6\tfinance\tmarkets\tTitle 6\tAbstract 6\t[]\t[]\n",
        encoding="utf-8",
    )
    (root / "behaviors.tsv").write_text(
        "1\tU1\t11/15/2019 9:55:12 AM\tN1 N2\tN3-1 N4-0 N5-0 N6-0\n"
        "2\tU2\t11/15/2019 10:00:00 AM\tN2\tN3-0 N4-0 N5-0 N6-0\n",
        encoding="utf-8",
    )


def test_mind_adapter_uses_logged_impression_candidates_only(tmp_path):
    _write_mind(tmp_path)
    adapter = MINDAdapter()
    first = list(
        adapter.build_instances(
            tmp_path,
            users=1,
            candidate_set_size=3,
            max_history_items=2,
            seed=2027,
        )
    )
    second = list(
        adapter.build_instances(
            tmp_path,
            users=1,
            candidate_set_size=3,
            max_history_items=2,
            seed=2027,
        )
    )
    assert first == second
    assert len(first) == 1
    instance = first[0]
    assert instance.demographics == {}
    assert instance.relevant_item_ids == frozenset({"N3"})
    assert "N3" in instance.candidate_ids()
    assert set(instance.candidate_ids()).issubset({"N3", "N4", "N5", "N6"})
    assert len(instance.candidates) == 3
    assert [item.item_id for item in instance.history] == ["N1", "N2"]
    assert instance.instance_metadata["identity_status"] == (
        "no_observed_demographics_synthetic_stress_only"
    )

    manifest = adapter.preprocessing_manifest()
    assert manifest["behavior_rows_seen"] == 2
    assert manifest["skipped_no_positive"] == 1
    assert manifest["emitted_tasks"] == 1


def test_mind_card_forbids_observed_demographic_claims():
    card = MINDAdapter().card()
    assert card.track == "synthetic_stress_test"
    assert "logged_impressions" in card.observed_fields
    assert "synthetic_identity_cue_only" in card.counterfactual_fields
    assert "does not provide observed demographic" in card.notes
