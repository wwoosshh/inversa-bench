from inversa.analysis import spearman


def test_perfect_positive():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_perfect_negative():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_no_variance_returns_none():
    assert spearman([1, 2, 3], [5, 5, 5]) is None


def test_too_few_points_returns_none():
    assert spearman([1], [2]) is None
