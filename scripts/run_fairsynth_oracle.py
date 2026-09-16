from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from faireval.conditions import build_personality_derangement
from faireval.datasets.fairsynth360 import FairSynth360Adapter
from faireval.freeze import canonical_json
from faireval.metrics import ndcg_at_k
from faireval.schema import OCEAN_KEYS, PersonalityProfile, UserInstance


DEFAULT_SEED = 1729
DEFAULT_USERS = 360
DEFAULT_CANDIDATE_SET_SIZE = 30
DEFAULT_HISTORY_ITEMS = 8
DEFAULT_K = 10
FEATURE_NAMES = ("exploration", "structure", "social", "calm", "novelty")


def _proxy_preference(profile: PersonalityProfile) -> tuple[float, ...]:
    """Expected latent preference from OCEAN under the FairSynth-v1 generator.

    FairSynth adds independent random latent components to the true preference
    vector. This proxy replaces each independent U(0,1) component by its expected
    value 0.5. It is therefore a deterministic personality-only oracle sanity
    model, not an LLM result and not a real-world psychometric claim.
    """
    values = profile.as_dict()
    o = values["openness"]
    c = values["conscientiousness"]
    e = values["extraversion"]
    a = values["agreeableness"]
    n = values["neuroticism"]
    return (
        0.65 * o + 0.20 * e + 0.15 * 0.5,
        0.65 * c + 0.20 * a + 0.15 * 0.5,
        0.65 * e + 0.20 * a + 0.15 * 0.5,
        0.55 * (1.0 - n) + 0.25 * c + 0.20 * 0.5,
        0.55 * o + 0.25 * (1.0 - n) + 0.20 * 0.5,
    )


def _rank(instance: UserInstance, preference: tuple[float, ...]) -> list[str]:
    def score(item) -> float:
        features = tuple(float(item.metadata[name]) for name in FEATURE_NAMES)
        return sum(a * b for a, b in zip(preference, features, strict=True)) / len(FEATURE_NAMES)

    return [
        item.item_id
        for item in sorted(instance.candidates, key=lambda item: (-score(item), item.item_id))
    ]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def run_oracle_sanity(
    *,
    users: int = DEFAULT_USERS,
    candidate_set_size: int = DEFAULT_CANDIDATE_SET_SIZE,
    max_history_items: int = DEFAULT_HISTORY_ITEMS,
    seed: int = DEFAULT_SEED,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    adapter = FairSynth360Adapter(relevant_items=5)
    instances = list(
        adapter.build_instances(
            Path("."),
            users=users,
            candidate_set_size=candidate_set_size,
            max_history_items=max_history_items,
            seed=seed,
        )
    )
    if len(instances) != users:
        raise RuntimeError(f"expected {users} instances, received {len(instances)}")
    by_id = {str(instance.user_id): instance for instance in instances}
    donor_map = build_personality_derangement(instances, seed=seed)

    true_ndcg: list[float] = []
    shuffled_ndcg: list[float] = []
    pva: list[float] = []
    identity_rank_hashes: list[str] = []

    for instance in instances:
        if instance.personality is None:
            raise RuntimeError(f"{instance.user_id} unexpectedly lacks synthetic OCEAN")
        true_ranking = _rank(instance, _proxy_preference(instance.personality))
        donor = by_id[donor_map[str(instance.user_id)]]
        if donor.personality is None:
            raise RuntimeError(f"donor {donor.user_id} unexpectedly lacks synthetic OCEAN")
        shuffled_ranking = _rank(instance, _proxy_preference(donor.personality))

        true_score = ndcg_at_k(true_ranking, instance.relevant_item_ids, k)
        shuffled_score = ndcg_at_k(shuffled_ranking, instance.relevant_item_ids, k)
        true_ndcg.append(float(true_score))
        shuffled_ndcg.append(float(shuffled_score))
        pva.append(float(true_score - shuffled_score))

        # The oracle ranker has no identity input. Re-labelling A/B/C therefore
        # cannot change this ranking by construction. Persist its hash as a
        # structural audit rather than fabricating three redundant rankings.
        identity_rank_hashes.append(_sha256_json(true_ranking[:k]))

    group_counts = Counter(
        str(instance.demographics["synthetic_identity_group"]) for instance in instances
    )
    result: dict[str, Any] = {
        "schema_version": "faireval-fairsynth-oracle-sanity-v1",
        "scope": "deterministic_non_llm_structural_validation_only",
        "dataset": "fairsynth360",
        "users": users,
        "candidate_set_size": candidate_set_size,
        "history_items": max_history_items,
        "relevant_items": 5,
        "k": k,
        "seed": seed,
        "identity_group_counts": dict(sorted(group_counts.items())),
        "identity_oracle_counterfactual_cug": 0.0,
        "identity_oracle_counterfactual_rank_change": 0.0,
        "identity_ground_truth_invariant_by_construction": True,
        "personality_proxy": {
            "description": "conditional-expectation proxy from synthetic OCEAN; non-LLM sanity model",
            "mean_true_ndcg_at_k": statistics.fmean(true_ndcg),
            "mean_shuffled_ndcg_at_k": statistics.fmean(shuffled_ndcg),
            "mean_true_minus_shuffled_ndcg": statistics.fmean(pva),
            "median_true_minus_shuffled_ndcg": statistics.median(pva),
            "fraction_users_true_gt_shuffled": sum(value > 0 for value in pva) / len(pva),
        },
        "guards": {
            "real_world_fairness_claim_allowed": False,
            "human_personality_claim_allowed": False,
            "llm_result": False,
            "confirmatory_p_value": False,
        },
        "trait_order": list(OCEAN_KEYS),
        "unique_identity_invariant_topk_hashes": len(set(identity_rank_hashes)),
    }
    result["artifact_sha256_without_self"] = _sha256_json(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the zero-cost deterministic FairSynth-360 oracle sanity experiment"
    )
    parser.add_argument("--output", type=Path, default=Path("results/smoke/fairsynth_oracle_v1.json"))
    parser.add_argument("--users", type=int, default=DEFAULT_USERS)
    parser.add_argument("--candidate-set-size", type=int, default=DEFAULT_CANDIDATE_SET_SIZE)
    parser.add_argument("--history-items", type=int, default=DEFAULT_HISTORY_ITEMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    result = run_oracle_sanity(
        users=args.users,
        candidate_set_size=args.candidate_set_size,
        max_history_items=args.history_items,
        seed=args.seed,
        k=args.k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
