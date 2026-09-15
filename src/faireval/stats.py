from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class PairedEffectSummary:
    n: int
    mean_difference: float
    median_difference: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PairedPermutationResult:
    n: int
    observed_mean_difference: float
    p_value: float
    alternative: str
    method: str
    permutations: int


def _paired_arrays(left: Sequence[float], right: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    if a.ndim != 1 or len(a) == 0:
        raise ValueError("paired arrays must be non-empty 1D sequences")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired arrays must contain only finite values")
    return a, b


def paired_bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2027,
) -> PairedEffectSummary:
    """User-level paired bootstrap for matched-condition mean differences.

    Resampling happens over paired users rather than over individual condition
    observations. This preserves the within-user counterfactual structure.
    """
    a, b = _paired_arrays(left, right)
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


def _is_as_or_more_extreme(
    permuted: np.ndarray,
    observed: float,
    *,
    alternative: Literal["two-sided", "greater", "less"],
) -> np.ndarray:
    tolerance = 1e-15
    if alternative == "two-sided":
        return np.abs(permuted) >= abs(observed) - tolerance
    if alternative == "greater":
        return permuted >= observed - tolerance
    return permuted <= observed + tolerance


def paired_permutation_test(
    left: Sequence[float],
    right: Sequence[float],
    *,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    exact_max_n: int = 18,
    samples: int = 100_000,
    seed: int = 2027,
) -> PairedPermutationResult:
    """Paired sign-flip permutation test on the mean condition difference.

    Under the paired null, each user's signed difference is exchangeable with its
    negative. For at most ``exact_max_n`` non-zero differences we enumerate every
    sign assignment exactly. Larger samples use Monte Carlo sign flips with the
    standard +1 correction, making the reported p-value deterministic for a
    frozen seed while avoiding an exponential memory/runtime blow-up.

    Zero differences are dropped from the sign-flip dimension because changing
    their sign cannot alter the statistic; ``n`` still records the full paired
    sample size.
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if exact_max_n < 0:
        raise ValueError("exact_max_n must be non-negative")
    if samples <= 0:
        raise ValueError("samples must be positive")

    a, b = _paired_arrays(left, right)
    diff = a - b
    observed = float(diff.mean())
    nonzero = diff[diff != 0.0]

    if len(nonzero) == 0:
        return PairedPermutationResult(
            n=len(diff),
            observed_mean_difference=observed,
            p_value=1.0,
            alternative=alternative,
            method="exact_all_zero",
            permutations=1,
        )

    # The test statistic must stay on the full-pair scale. Zeros contribute zero
    # to every permutation, so the denominator remains the original paired n.
    denominator = float(len(diff))

    if len(nonzero) <= exact_max_n:
        total = 1 << len(nonzero)
        indices = np.arange(total, dtype=np.uint64)[:, None]
        bit_positions = np.arange(len(nonzero), dtype=np.uint64)[None, :]
        bits = ((indices >> bit_positions) & 1).astype(np.int8)
        signs = 1.0 - 2.0 * bits
        permuted_means = (signs * nonzero[None, :]).sum(axis=1) / denominator
        extreme = _is_as_or_more_extreme(permuted_means, observed, alternative=alternative)
        p_value = float(extreme.mean())
        return PairedPermutationResult(
            n=len(diff),
            observed_mean_difference=observed,
            p_value=p_value,
            alternative=alternative,
            method="exact_sign_flip",
            permutations=total,
        )

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(nonzero)))
    permuted_means = (signs * nonzero[None, :]).sum(axis=1) / denominator
    extreme_count = int(
        _is_as_or_more_extreme(permuted_means, observed, alternative=alternative).sum()
    )
    p_value = (extreme_count + 1.0) / (samples + 1.0)
    return PairedPermutationResult(
        n=len(diff),
        observed_mean_difference=observed,
        p_value=float(p_value),
        alternative=alternative,
        method="monte_carlo_sign_flip",
        permutations=samples,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down family-wise-error adjusted p-values.

    The output order matches the input order. Values must be finite probabilities
    in [0, 1]. The monotonicity step is essential: adjusted p-values cannot become
    smaller as the sorted raw p-values become larger.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be a 1D sequence")
    if len(p) == 0:
        return []
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must be finite probabilities in [0,1]")

    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    m = len(sorted_p)
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, raw in enumerate(sorted_p):
        candidate = min(1.0, (m - rank) * float(raw))
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = running_max

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]
