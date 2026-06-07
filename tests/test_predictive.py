"""E14 graders (predictive validity). Y1 = constrained construction: the model outputs a polynomial
equation P(x)=0 and a sympy grader checks ALL exact constraints (degree, integer coeffs, a required
root, leading coefficient, number of distinct real roots). Machine-verified, no LLM judge. Tested on
known-pass / known-fail polynomials so the grader itself can't drift.
"""
from __future__ import annotations

from inversa.predictive import check_construction


# x**4 - 1 = (x-1)(x+1)(x^2+1): degree 4, integer coeffs, root at 1, leading coeff 1, 2 distinct reals
_CHECKS = [
    {"type": "degree", "value": 4},
    {"type": "integer_coeffs"},
    {"type": "root_at", "x": 1},
    {"type": "leading_coeff", "value": 1},
    {"type": "num_distinct_real_roots", "value": 2},
]


def test_construction_all_constraints_satisfied():
    ok, per = check_construction(_CHECKS, "x**4 - 1 = 0")
    assert ok is True
    assert all(p for _c, p in per)


def test_construction_wrong_degree_fails():
    ok, _ = check_construction(_CHECKS, "x**2 - 1 = 0")   # degree 2, not 4
    assert ok is False


def test_construction_missing_required_root_fails():
    ok, _ = check_construction(_CHECKS, "x**4 - 16 = 0")  # roots +/-2, not a root at 1
    assert ok is False


def test_construction_noninteger_coeffs_fails():
    checks = [{"type": "integer_coeffs"}]
    assert check_construction(checks, "x**2 - 1/2 = 0")[0] is False


def test_construction_distinct_real_root_count():
    # (x-1)^2 (x+1) = degree 3, but only 2 DISTINCT real roots (1 with multiplicity 2, and -1)
    ok, _ = check_construction([{"type": "num_distinct_real_roots", "value": 2}],
                               "(x-1)**2 * (x+1) = 0")
    assert ok is True
    ok2, _ = check_construction([{"type": "num_distinct_real_roots", "value": 3}],
                               "(x-1)**2 * (x+1) = 0")
    assert ok2 is False


def test_construction_unparseable_fails_gracefully():
    ok, _ = check_construction(_CHECKS, "this is not an equation")
    assert ok is False


def test_construction_unique_real_root():
    # x**3 + x - 2 = (x-1)(x^2+x+2): unique real root 1
    ok, _ = check_construction([{"type": "unique_real_root", "value": 1.0}], "x**3 + x - 2 = 0")
    assert ok is True
