from pathlib import Path

from faireval.datasets.fairsynth360 import (
    FairSynth360Adapter,
    synthetic_items,
    synthetic_users,
)


def test_fairsynth_identity_is_exactly_balanced():
    users = synthetic_users()
    counts = {group: sum(user.identity_group == group for user in users) for group in ("A", "B", "C")}
    assert counts == {"A": 120, "B": 120, "C": 120}


def test_fairsynth_catalog_and_user_counts_are_frozen():
    assert len(synthetic_users()) == 360
    assert len(synthetic_items()) == 120
    assert len({item.item_id for item in synthetic_items()}) == 120


def test_fairsynth_build_is_deterministic_and_valid():
    adapter_a = FairSynth360Adapter()
    adapter_b = FairSynth360Adapter()
    left = list(
        adapter_a.build_instances(
            Path("."), users=12, candidate_set_size=30, max_history_items=8, seed=1729
        )
    )
    right = list(
        adapter_b.build_instances(
            Path("."), users=12, candidate_set_size=30, max_history_items=8, seed=1729
        )
    )
    assert [row.user_id for row in left] == [row.user_id for row in right]
    assert [row.candidate_ids() for row in left] == [row.candidate_ids() for row in right]
    assert [row.relevant_item_ids for row in left] == [row.relevant_item_ids for row in right]
    for instance in left:
        instance.validate()
        assert len(instance.candidates) == 30
        assert len(instance.history) == 8
        assert len(instance.relevant_item_ids) == 5
        assert instance.instance_metadata["synthetic"] is True
        assert "synthetic_identity_group" in instance.demographics
        assert instance.personality is not None


def test_fairsynth_identity_not_used_in_ground_truth_metadata():
    instance = next(
        FairSynth360Adapter().build_instances(
            Path("."), users=1, candidate_set_size=30, max_history_items=8, seed=1729
        )
    )
    assert instance.instance_metadata["identity_semantics"] == (
        "balanced_semantically_meaningless_independent_of_relevance"
    )
    assert instance.instance_metadata["ground_truth_relevance_policy"] == (
        "top_latent_utility_within_candidate_pool"
    )
