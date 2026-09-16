from scripts.run_fairsynth_oracle import run_oracle_sanity


def test_fairsynth_oracle_is_deterministic_and_identity_invariant():
    first = run_oracle_sanity()
    second = run_oracle_sanity()
    assert first == second
    assert first["identity_group_counts"] == {"A": 120, "B": 120, "C": 120}
    assert first["identity_oracle_counterfactual_cug"] == 0.0
    assert first["identity_oracle_counterfactual_rank_change"] == 0.0
    assert first["identity_ground_truth_invariant_by_construction"] is True
    assert first["guards"]["llm_result"] is False
    assert first["guards"]["real_world_fairness_claim_allowed"] is False


def test_fairsynth_personality_proxy_has_positive_signal_in_v1():
    result = run_oracle_sanity()
    personality = result["personality_proxy"]
    assert personality["mean_true_ndcg_at_k"] > personality["mean_shuffled_ndcg_at_k"]
    assert personality["mean_true_minus_shuffled_ndcg"] > 0.0
