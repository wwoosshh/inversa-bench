"""Dependency-free Spearman rank correlation (suggestive with few points) + bootstrap CIs."""
from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple


def _avg_ranks(vals: Sequence[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((a[i] - ma) ** 2 for i in range(n))
    vb = sum((b[i] - mb) ** 2 for i in range(n))
    if va == 0 or vb == 0:
        return None
    return cov / ((va * vb) ** 0.5)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_avg_ranks(xs), _avg_ranks(ys))


def spearman_ci(xs: Sequence[float], ys: Sequence[float],
                n_boot: int = 2000, alpha: float = 0.05,
                seed: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Point estimate + percentile bootstrap (1-alpha) CI for Spearman rho. Seeded for
    reproducibility. Returns (rho, lo, hi); (None, None, None) if < 3 usable pairs. The CI is
    what turns a suggestive point estimate into a publishable, uncertainty-quantified claim."""
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    if len(pairs) < 3:
        return (None, None, None)
    base = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    rng = random.Random(seed)
    n = len(pairs)
    boots: List[float] = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        s = spearman([p[0] for p in sample], [p[1] for p in sample])
        if s is not None:
            boots.append(s)
    if len(boots) < 10:
        return (base, None, None)
    boots.sort()
    lo = boots[int((alpha / 2) * len(boots))]
    hi = boots[min(len(boots) - 1, int((1 - alpha / 2) * len(boots)))]
    return (base, lo, hi)
