"""Artifact fix (a): numeric fallback so valid transcendental constructions (which sympy
cannot symbolically solve) are accepted, while non-unique ones are still rejected.
Artifact fix (b): extract_equation honors a '#### <eq>' final-answer marker."""
import sympy as sp

from inversa.tasks.posing import extract_equation
from inversa.verifiers.math_equation import verify_equation

_X = sp.Symbol("x")


def test_numeric_validates_monotonic_exp_embedding():
    # exp(g(x)) = 1  <=>  g(x) = 0; g = x**3 - x - 1 has one real root -> unique.
    plastic = float(sp.nsolve(_X**3 - _X - 1, _X, 1.3))
    r = verify_equation("exp(x**3 - x - 1) = 1", plastic)
    assert r.valid is True
    assert r.unique is True


def test_numeric_validates_strictly_monotonic_pose():
    r = verify_equation("exp(x) + x = exp(3) + 3", 3)
    assert r.valid is True
    assert r.unique is True


def test_numeric_rejects_nonunique_transcendental():
    # (x**2 - 5)**2 = 24 has four real roots -> the target is a member but NOT unique.
    val = float(sp.sqrt(2) + sp.sqrt(3))
    r = verify_equation("exp((x**2 - 5)**2) = exp(24)", val)
    assert r.valid is True
    assert r.unique is False


def test_numeric_rejects_non_root():
    r = verify_equation("exp(x) + x = exp(3) + 3", 5)  # the root is 3, not 5
    assert r.valid is False


def test_polynomial_path_unaffected():
    assert verify_equation("x**2 - 4 = 0", 2).unique is False   # roots 2, -2
    assert verify_equation("(x - 3)*(x**2 + 1) = 0", 3).unique is True


def test_extract_prefers_marker_after_prose():
    raw = "Let me reason: substitute x -> x-1.\nSo the answer is\n#### (x-1)**3 + (x-1) - 1 = 0"
    assert extract_equation(raw) == "(x-1)**3 + (x-1) - 1 = 0"


def test_extract_last_equation_line_when_no_marker():
    raw = "x + 1 = 2 (a guess)\nx**2 - 4 = 0"
    assert extract_equation(raw) == "x**2 - 4 = 0"
