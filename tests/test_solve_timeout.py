"""sp.solve has no native timeout and can hang forever on adversarial model output
(this stalled a multi-model run for ~46 min). verify_equation now bounds every solve."""
import time

import pytest
import sympy as sp

import inversa.verifiers.math_equation as me


def test_solve_with_timeout_returns_promptly_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(me.sp, "solve", lambda *a, **k: time.sleep(30))
    start = time.time()
    with pytest.raises(TimeoutError):
        me._solve_with_timeout(sp.Eq(me._X - 1, 0), me._X, timeout=0.3)
    assert time.time() - start < 5  # did NOT wait the full 30s


def test_solve_with_timeout_normal_result_unaffected():
    sols = me._solve_with_timeout(sp.Eq(me._X**2 - 4, 0), me._X, timeout=5)
    assert {int(s) for s in sols} == {-2, 2}
