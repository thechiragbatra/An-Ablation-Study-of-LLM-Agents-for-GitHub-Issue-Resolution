"""Statistics for paired variant comparisons. Pure Python, no scipy."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class McNemarResult:
    both: int  # resolved by both
    a_only: int  # resolved by A, not B
    b_only: int  # resolved by B, not A
    neither: int
    p_value: float  # exact two-sided

    @property
    def n(self) -> int:
        return self.both + self.a_only + self.b_only + self.neither


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: binomial test of b vs c discordant pairs at p=0.5."""
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def mcnemar(a: Mapping[str, bool], b: Mapping[str, bool]) -> McNemarResult:
    """Paired comparison over the instances present in both mappings."""
    ids = sorted(a.keys() & b.keys())
    both = sum(a[i] and b[i] for i in ids)
    a_only = sum(a[i] and not b[i] for i in ids)
    b_only = sum(b[i] and not a[i] for i in ids)
    neither = len(ids) - both - a_only - b_only
    return McNemarResult(both, a_only, b_only, neither, mcnemar_exact_p(a_only, b_only))


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def majority(outcomes: Sequence[bool]) -> bool:
    """Strict majority (e.g. at least 2 of 3 seeds)."""
    return sum(outcomes) * 2 > len(outcomes)


def bootstrap_ci(
    units: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | None = None,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI over independent units (instances)."""
    if not units:
        raise ValueError("no units to bootstrap")
    stat = statistic or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    n = len(units)
    draws = sorted(stat([units[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    lo = draws[math.floor((alpha / 2) * (n_resamples - 1))]
    hi = draws[math.ceil((1 - alpha / 2) * (n_resamples - 1))]
    return lo, hi


def paired_diff_ci(
    a: Mapping[str, float],
    b: Mapping[str, float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(mean(b - a), lo, hi) over shared instances, resampling instances jointly."""
    ids = sorted(a.keys() & b.keys())
    diffs = [b[i] - a[i] for i in ids]
    lo, hi = bootstrap_ci(diffs, n_resamples=n_resamples, alpha=alpha, seed=seed)
    return sum(diffs) / len(diffs), lo, hi


def mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def stdev(xs: Sequence[float]) -> float | None:
    """Sample standard deviation (n-1); None for fewer than 2 values."""
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
