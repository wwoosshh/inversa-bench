"""E14 predictive-validity graders (machine-verified, no LLM judge).

Y1 (constrained construction): the model outputs a polynomial equation P(x)=0; `check_construction`
parses P = lhs - rhs in the sandboxed namespace and checks every exact constraint with sympy. This
loads on the construction / self-verification residual of §3.7 (build an object meeting several exact
specs at once), distinct from forward solving. Y2 (error detection) needs no grader here — its claims'
truth is precomputed by sympy at bank-build time, so scoring is exact yes/no matching.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import sympy as sp

from inversa.verifiers.math_equation import parse_sides

_X = sp.Symbol("x")


def _poly(equation_str: str):
    """P(x) = lhs - rhs as a sympy Poly in x, or None if it doesn't parse as a polynomial in x."""
    try:
        lhs, rhs = parse_sides(equation_str)
    except Exception:
        return None
    expr = sp.expand(lhs - rhs)
    if _X not in expr.free_symbols:
        return None
    try:
        return sp.Poly(expr, _X)
    except Exception:
        return None


def _distinct_real_roots(poly):
    try:
        return len(set(poly.real_roots()))   # real_roots lists with multiplicity; set -> distinct
    except Exception:
        return None


def _check_one(check: Dict[str, Any], poly) -> bool:
    t = check["type"]
    try:
        if t == "degree":
            return poly.degree() == check["value"]
        if t == "integer_coeffs":
            return all(c.is_Integer for c in poly.all_coeffs())
        if t == "leading_coeff":
            return sp.nsimplify(poly.LC()) == check["value"]
        if t == "root_at":
            return abs(float(poly.eval(check["x"]))) < 1e-9
        if t == "num_distinct_real_roots":
            return _distinct_real_roots(poly) == check["value"]
        if t == "unique_real_root":
            roots = poly.real_roots()
            return len(set(roots)) == 1 and abs(float(roots[0]) - float(check["value"])) < 1e-6
    except Exception:
        return False
    return False  # unknown check type -> fail closed


def check_construction(checks: Sequence[Dict[str, Any]],
                       equation_str: str) -> Tuple[bool, List[Tuple[Dict[str, Any], bool]]]:
    """Grade a constrained-construction answer: return (all_constraints_pass, per-check results).
    An unparseable / non-polynomial answer fails every check (no credit for non-answers)."""
    poly = _poly(equation_str)
    if poly is None:
        return (False, [(c, False) for c in checks])
    per = [(c, _check_one(c, poly)) for c in checks]
    return (all(p for _c, p in per), per)
