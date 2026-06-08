"""Anti-gaming for pose (rebuttal #5): `x - target = 0` (and any linear-in-x equation) merely
*restates* the answer — it is not construction. The verifier flags such trivial restatements, and
the pose runner can be told to reject them (require_nontrivial), so a model must build real
structure (degree >= 2, or a non-polynomial form) to score. Default stays permissive for
back-compatibility with the existing leaderboard; the strict mode is opt-in.
"""
from __future__ import annotations

import math

import sympy as sp

from inversa.adapters.fake import FakeAdapter
from inversa.tasks.structural import run_struct_pose_item
from inversa.verifiers.math_equation import verify_equation


def test_linear_equation_is_flagged_trivial():
    r = verify_equation("x - 3 = 0", 3.0)
    assert r.valid and r.unique and r.trivial            # correct answer, but pure restatement


def test_scaled_linear_still_trivial():
    assert verify_equation("2*x - 6 = 0", 3.0).trivial    # 2x-6=0 is still just x=3


def test_literal_irrational_restatement_is_trivial():
    v = float(sp.sqrt(2) + 1)
    assert verify_equation("x = sqrt(2) + 1", v).trivial   # restating the target literal


def test_genuine_construction_is_not_trivial():
    r = verify_equation("x**3 - 27 = 0", 3.0)              # unique real root 3, built with structure
    assert r.valid and r.unique and not r.trivial


def test_nonpolynomial_is_not_flagged_trivial():
    # strictly-increasing cubic pinning sqrt(2) as the UNIQUE real root — real construction
    v = float(sp.sqrt(2))
    rhs = v ** 3 + v
    r = verify_equation(f"x**3 + x - {rhs} = 0", v)
    assert r.valid and r.unique and not r.trivial


def test_runner_default_accepts_trivial_for_backcompat():
    ad = FakeAdapter(["#### x - 3 = 0"])
    it = run_struct_pose_item(ad, "3", 3.0, "integer")
    assert it.valid                                        # default: trivial still counts


def test_runner_strict_mode_rejects_trivial():
    ad = FakeAdapter(["#### x - 3 = 0"])
    it = run_struct_pose_item(ad, "3", 3.0, "integer", require_nontrivial=True)
    assert not it.valid                                    # strict: restatement does not count


def test_runner_strict_mode_accepts_real_construction():
    ad = FakeAdapter(["#### x**3 - 27 = 0"])
    it = run_struct_pose_item(ad, "3", 3.0, "integer", require_nontrivial=True)
    assert it.valid


def test_oversized_equation_is_rejected_not_parsed():
    # adversarial model output: an enormous string should be refused before sympy parses it
    huge = "x" + "+1" * 5000 + " = 0"
    r = verify_equation(huge, 0.0)
    assert r.well_formed is False and "too long" in (r.error or "")
