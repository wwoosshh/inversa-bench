"""Level-3 probes: construction/transformation of structure, not template recall.
Verifies the recall-proof core — uniqueness breaks naive templating (Form A) and a
structural substitution propagates an ugly root (Form B)."""
import sympy as sp

from inversa.adapters.fake import FakeAdapter
from inversa.tasks.structural import (
    build_struct_pose_prompt,
    build_transform_prompt,
    run_struct_pose_item,
    run_transform_item,
    validity_rate,
)
from inversa.verifiers.math_equation import verify_equation

_X = sp.Symbol("x")


def test_struct_pose_prompt_mentions_target_and_uniqueness():
    p = build_struct_pose_prompt("sqrt(2) + 1")
    assert "sqrt(2) + 1" in p and "UNIQUE" in p


def test_struct_pose_unique_irrational_is_valid():
    val = float(sp.sqrt(2) + 1)
    poser = FakeAdapter(["x = sqrt(2) + 1"])  # trivially the unique real solution
    item = run_struct_pose_item(poser, "sqrt(2) + 1", val, novelty="quadratic-irrational")
    assert item.valid is True
    assert item.novelty == "quadratic-irrational"


def test_minimal_polynomial_fails_uniqueness():
    # The naive "template" for sqrt(2)+1 is its minimal polynomial x**2 - 2x - 1 = 0,
    # but that ALSO has the conjugate root 1 - sqrt(2) -> NOT unique -> level-3 failure.
    val = float(sp.sqrt(2) + 1)
    poser = FakeAdapter(["x**2 - 2*x - 1 = 0"])
    item = run_struct_pose_item(poser, "sqrt(2) + 1", val)
    assert item.verification.valid is True   # sqrt(2)+1 IS a root (membership)
    assert item.valid is False               # but not the unique real one


def test_transform_prompt_mentions_source_and_g():
    p = build_transform_prompt("x**3 + x - 1 = 0", "r + 1")
    assert "x**3 + x - 1 = 0" in p and "r + 1" in p


def test_transform_structural_substitution_is_valid():
    # E = x**3 + x - 1 = 0 has one real (ugly) root r ~= 0.6823. The structural answer for
    # g(r) = r + 1 is the substitution x -> x - 1: (x-1)**3 + (x-1) - 1 = 0.
    r = float(sp.nsolve(_X**3 + _X - 1, _X, 0.7))
    target = r + 1.0
    poser = FakeAdapter(["(x - 1)**3 + (x - 1) - 1 = 0"])
    item = run_transform_item(poser, "x**3 + x - 1 = 0", "r + 1", r, target)
    assert item.valid is True
    assert abs(item.target_value - target) < 1e-9


def test_transform_numeric_template_with_wrong_value_fails():
    r = float(sp.nsolve(_X**3 + _X - 1, _X, 0.7))
    target = r + 1.0
    poser = FakeAdapter(["x = 5"])   # ignored the structure, wrong value
    item = run_transform_item(poser, "x**3 + x - 1 = 0", "r + 1", r, target)
    assert item.valid is False


def test_validity_rate_counts_unique_successes():
    val = float(sp.sqrt(2) + 1)
    poser = FakeAdapter(["x = sqrt(2) + 1", "x**2 - 2*x - 1 = 0"])  # one unique, one not
    items = [run_struct_pose_item(poser, "sqrt(2) + 1", val) for _ in range(2)]
    assert validity_rate(items) == 0.5


def test_relational_output_is_malformed_not_crash():
    # regression: a model emitting sympy relational syntax must be rejected cleanly,
    # not crash later arithmetic with "Equality - Zero".
    r = verify_equation("Eq(x**2, 4) = 0", 2)
    assert r.well_formed is False
    assert r.valid is False
