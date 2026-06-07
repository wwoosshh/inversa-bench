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


# --- partial Spearman (discriminant validity: control general capability) ----

def test_partial_spearman_matches_standard_formula():
    # contract: rank partial correlation = (r_xy - r_xz r_yz)/sqrt((1-r_xz^2)(1-r_yz^2))
    from inversa.analysis import spearman, partial_spearman
    x = [3, 1, 4, 1, 5, 9, 2, 6]
    y = [2, 7, 1, 8, 2, 8, 1, 9]
    z = [1, 4, 1, 5, 9, 2, 6, 5]
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    expected = (rxy - rxz * ryz) / (((1 - rxz ** 2) * (1 - ryz ** 2)) ** 0.5)
    assert abs(partial_spearman(x, y, z) - expected) < 1e-9


def test_partial_spearman_drops_correlation_explained_by_control():
    # x and y are correlated ONLY because both follow z; controlling z collapses it toward 0
    from inversa.analysis import spearman, partial_spearman
    z = [1, 2, 3, 4, 5, 6, 7, 8]
    x = [2, 1, 3, 4, 5, 6, 7, 8]          # x tracks z (one early swap, not identical)
    y = [1, 2, 3, 4, 5, 6, 8, 7]          # y tracks z too (one late swap, not identical)
    assert spearman(x, y) > 0.9            # strong raw correlation
    assert partial_spearman(x, y, z) < 0.5  # mostly explained away by z


def test_partial_spearman_degenerate_returns_none():
    from inversa.analysis import partial_spearman
    # control identical to a variable -> denominator 0 -> undefined
    assert partial_spearman([1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4]) is None
