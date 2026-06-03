"""Machine difficulty proxy for a posed equation: polynomial degree in x (v1).

band = polynomial degree of (lhs - rhs) in x, clamped to 1..5; non-polynomial
(transcendental) equations are treated as maximum difficulty (band 5). band 0
means "not measurable" (parse failure or the equation reduces to an identity).
This is a transparent v1 proxy — richer signals (operation count, solution
class, solver step-count) are future work (design §12). Glass-box: degree and
the non-polynomial flag are returned as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp

from inversa.verifiers.math_equation import parse_sides

_X = sp.Symbol("x")


@dataclass
class DifficultyResult:
    band: int                   # 1..5; 0 if not measurable
    degree: Optional[int]       # polynomial degree in x, or None if non-polynomial
    non_polynomial: bool
    error: Optional[str] = None


def difficulty(equation_str: str) -> DifficultyResult:
    try:
        lhs, rhs = parse_sides(equation_str)
    except ValueError as e:
        return DifficultyResult(0, None, False, str(e))

    expr = sp.expand(lhs - rhs)
    if expr == 0:
        return DifficultyResult(0, None, False, "identity (x cancels; no constraint)")

    try:
        deg = int(sp.degree(expr, gen=_X))
    except (sp.PolynomialError, TypeError, ValueError):
        return DifficultyResult(5, None, True)

    if deg <= 0:
        return DifficultyResult(0, deg, False, "no x term")
    return DifficultyResult(max(1, min(5, deg)), deg, False)
