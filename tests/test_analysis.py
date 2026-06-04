from inversa.analysis import spearman


def test_perfect_positive():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_perfect_negative():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_no_variance_returns_none():
    assert spearman([1, 2, 3], [5, 5, 5]) is None


def test_too_few_points_returns_none():
    assert spearman([1], [2]) is None


def test_tie_averaging_is_handled():
    # ties must not crash and must still yield a valid correlation
    r = spearman([1, 2, 2, 4], [10, 20, 30, 40])
    assert r is not None and 0.0 < r <= 1.0


def test_spearman_ci_perfect_monotonic():
    from inversa.analysis import spearman_ci
    rho, lo, hi = spearman_ci([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], n_boot=500)
    assert rho == 1.0
    assert lo is not None and hi is not None and lo <= 1.0 <= hi + 1e-9


def test_spearman_ci_too_few_points():
    from inversa.analysis import spearman_ci
    assert spearman_ci([1, 2], [2, 1]) == (None, None, None)
