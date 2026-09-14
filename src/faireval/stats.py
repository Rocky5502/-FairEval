from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PairedEffectSummary:
    n: int
    mean_difference: float
    median_difference: float
    ci_low: float
    ci_high: float


def paired_bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2027,
) -> PairedEffectSummary:
    """User-level paired bootstrap for matched-condition differences."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    if a.ndim != 1 or len(a) == 0:
        raise ValueError("paired arrays must be non-empty 1D sequences")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")

    diff = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff), size=(samples, len(diff)))
    bootstrap_means = diff[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return PairedEffectSummary(
        n=len(diff),
        mean_difference=float(diff.mean()),
        median_difference=float(np.median(diff)),
        ci_low=float(low),
        ci_high=float(high),
    )
