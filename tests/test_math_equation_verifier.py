from inversa.verifiers.math_equation import verify_equation


def test_linear_equation_valid_and_unique():
    r = verify_equation("2*x + 1 = 7", 3)
    assert r.well_formed is True
    assert r.valid is True
    assert r.unique is True
    assert r.error is None


def test_implicit_multiplication_parsed():
    r = verify_equation("2x = 6", 3)
    assert r.well_formed is True
    assert r.valid is True
    assert r.unique is True


def test_quadratic_valid_but_not_unique():
    r = verify_equation("x**2 = 9", 3)
    assert r.well_formed is True
    assert r.valid is True
    assert r.unique is False  # solutions are 3 and -3


def test_target_zero():
    r = verify_equation("x = 0", 0)
    assert r.valid is True and r.unique is True


def test_wrong_target_is_invalid():
    r = verify_equation("2*x = 6", 5)
    assert r.well_formed is True
    assert r.valid is False


def test_no_variable_is_not_well_formed():
    r = verify_equation("2 + 2 = 4", 4)
    assert r.well_formed is False
    assert r.valid is False
    assert "x" in r.error


def test_missing_equals_is_not_well_formed():
    r = verify_equation("2*x + 1", 3)
    assert r.well_formed is False
    assert r.valid is False


def test_parse_error_is_not_well_formed():
    r = verify_equation("x + = 3", 3)
    assert r.well_formed is False
    assert r.valid is False
    assert r.error is not None


def test_inequality_rejected():
    for s in ("x != 3", "x <= 3", "x >= 3"):
        r = verify_equation(s, 3)
        assert r.well_formed is False


def test_identity_not_well_formed():
    r = verify_equation("x = x", 3)
    assert r.well_formed is False
    assert "x" in r.error


def test_code_injection_is_blocked():
    # untrusted model output must not execute arbitrary Python; must be rejected safely
    r = verify_equation('__import__("os").getcwd() = 1', 1)
    assert r.well_formed is False
    assert r.valid is False


def test_unsolvable_equation_is_well_formed_but_invalid():
    r = verify_equation("cos(x) = x", 1)
    assert r.well_formed is True
    assert r.valid is False
