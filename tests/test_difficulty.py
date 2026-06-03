from inversa.verifiers.difficulty import difficulty


def test_linear_is_band_1():
    r = difficulty("2*x + 1 = 7")
    assert r.band == 1
    assert r.degree == 1
    assert r.non_polynomial is False


def test_quadratic_is_band_2():
    r = difficulty("x**2 - 4 = 0")
    assert r.band == 2 and r.degree == 2


def test_cubic_is_band_3():
    r = difficulty("x**3 - x = 0")
    assert r.band == 3 and r.degree == 3


def test_high_degree_clamped_to_5():
    r = difficulty("x**7 = 1")
    assert r.band == 5 and r.degree == 7


def test_non_polynomial_is_band_5():
    r = difficulty("cos(x) = x")
    assert r.band == 5 and r.non_polynomial is True


def test_identity_is_band_0():
    r = difficulty("x = x")
    assert r.band == 0 and r.error is not None


def test_unparseable_is_band_0():
    r = difficulty("x + = 3")
    assert r.band == 0 and r.error is not None


def test_constant_no_x_term_is_band_0():
    r = difficulty("x**2 - x**2 = 3")
    assert r.band == 0
    assert r.degree == 0
    assert "no x term" in r.error
