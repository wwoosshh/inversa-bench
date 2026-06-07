"""Randomized pose-target generator — closes the pose-contamination gap (weakness #3).

The original pose task drew targets from a FIXED bank; once that bank is public, the
(target -> good equation) pairs can leak and be memorized. Here targets are generated FRESH at
test time across the same novelty ladder (irrational / structural values), with random parameters,
so there is no fixed item to leak — the same recall-resistance the transform task gets from
randomized inputs.

Correctness is by construction: each target's numeric value is computed by sympy FROM the same
symbolic object that renders its description, so the description and the value cannot drift apart
(asserted in tests). The generator is deterministic given a `random.Random`, so a run can record
its seed and be reproduced/audited, while different seeds give different targets.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

import sympy as sp

_T = sp.Symbol("t")


def _nonsquare(rng: random.Random) -> int:
    while True:
        a = rng.randint(2, 40)
        if sp.sqrt(a) != int(sp.sqrt(a)):  # not a perfect square -> sqrt(a) irrational
            return a


def _noncube(rng: random.Random) -> int:
    while True:
        a = rng.randint(2, 40)
        if round(a ** (1 / 3)) ** 3 != a:  # not a perfect cube
            return a


def _quadratic_irrational(rng: random.Random):
    a = _nonsquare(rng)
    form = rng.choice(["shift_plus", "shift_minus", "scale", "halfsum"])
    if form == "shift_plus":
        b = rng.randint(1, 9)
        return sp.sqrt(a) + b, "quadratic-irrational"
    if form == "shift_minus":
        b = rng.randint(1, 9)
        return b - sp.sqrt(a), "quadratic-irrational"
    if form == "scale":
        c = rng.randint(2, 5)
        return c * sp.sqrt(a), "quadratic-irrational"
    b, c = rng.randint(1, 9), rng.choice([2, 3])
    return (b + sp.sqrt(a)) / c, "quadratic-irrational"


def _sum_of_surds(rng: random.Random):
    a = _nonsquare(rng)
    b = _nonsquare(rng)
    while b == a:
        b = _nonsquare(rng)
    return sp.sqrt(a) + sp.sqrt(b), "sum-of-surds"


def _nested_radical(rng: random.Random):
    a = rng.randint(1, 6)
    b = _nonsquare(rng)
    return sp.sqrt(a + sp.sqrt(b)), "nested-radical"


def _cube_root(rng: random.Random):
    a = _noncube(rng)
    return sp.Rational(a) ** sp.Rational(1, 3), "cube-root"


def _rational(rng: random.Random):
    q = rng.randint(2, 9)
    p = rng.randint(q + 1, 3 * q)  # > 1, not an integer multiple in most cases
    while p % q == 0:
        p = rng.randint(q + 1, 3 * q)
    return sp.Rational(p, q), "rational"


def _closed_form(rng: random.Random):
    """A target with a closed-form symbolic description (desc = sympy string of the value)."""
    gen = rng.choice([_quadratic_irrational, _sum_of_surds, _nested_radical,
                      _cube_root, _rational])
    expr, novelty = gen(rng)
    return sp.sstr(expr), float(expr), novelty


def _cubic_irrational(rng: random.Random):
    """A structurally-described target: the UNIQUE real root of t**3 + p*t + q (p>0 => strictly
    increasing => exactly one real root, so 'the real root' is well-defined)."""
    p = rng.randint(1, 6)
    q = rng.choice([-1, 1]) * rng.randint(1, 9)
    qs = f"+ {q}" if q >= 0 else f"- {abs(q)}"
    desc = f"the real root of t**3 + {p}*t {qs} = 0"
    value = float(sp.nsolve(_T ** 3 + p * _T + q, _T, 1.0))
    return desc, value, "cubic-irrational"


def random_pose_targets(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Generate `n` fresh pose targets (desc/value/novelty). Mixes closed-form irrationals with
    structurally-defined cubic roots so the set spans the novelty ladder and is not memorizable."""
    targets: List[Dict[str, Any]] = []
    for _ in range(n):
        builder = _cubic_irrational if rng.random() < 0.25 else _closed_form
        desc, value, novelty = builder(rng)
        targets.append({"target_desc": desc, "target_value": value, "novelty": novelty})
    return targets
