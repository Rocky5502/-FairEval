from __future__ import annotations

import hashlib
import random
from dataclasses import replace

from .schema import UserInstance


def _stable_seed(instance: UserInstance, order_seed: int) -> int:
    raw = f"{instance.dataset}|{instance.user_id}|{order_seed}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def permute_candidate_order(instance: UserInstance, *, order_seed: int) -> UserInstance:
    """Deterministically permute candidate order for an entire paired experiment.

    Call this once per user instance/order seed, then reuse the returned instance
    for every demographic/personality condition in that pair. This preserves the
    candidate set while making order a controlled nuisance factor rather than a
    hidden confound.
    """
    instance.validate()
    candidates = list(instance.candidates)
    rng = random.Random(_stable_seed(instance, order_seed))
    rng.shuffle(candidates)
    return replace(instance, candidates=tuple(candidates))
