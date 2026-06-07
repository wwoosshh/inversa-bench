"""IRT scaling (rebuttal #6): IGS as a raw mean over a convenience bank is only ordinal — a 0.9
vs 0.8 gap is not metrically meaningful, and pose/transform are averaged with arbitrary equal
weight. A Rasch (1PL) fit over the model x item correctness matrix puts model ability on an
interval (logit) theta scale and gives each item a difficulty; point-biserial gives item
discrimination. These are pure, dependency-free, and tested on structured matrices.
"""
from __future__ import annotations

from inversa.irt import point_biserial, rasch_fit


# A Guttman staircase: model j is correct on items 0..j -> a clean, perfectly Rasch-orderable matrix.
_GUTTMAN = [
    [1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
]


def test_theta_increases_with_raw_score():
    theta, _b = rasch_fit(_GUTTMAN)
    assert theta == sorted(theta)              # higher raw score -> higher ability
    assert theta[0] < theta[-1]


def test_difficulty_increases_as_fewer_models_pass():
    _theta, b = rasch_fit(_GUTTMAN)
    # item 0 is passed by everyone (easiest), item 4 by only the best (hardest)
    assert b == sorted(b)
    assert b[0] < b[-1]


def test_difficulties_are_centered_for_identifiability():
    _theta, b = rasch_fit(_GUTTMAN)
    assert abs(sum(b) / len(b)) < 1e-6


def test_theta_is_interval_not_just_rank():
    # equal raw-score *gaps* need not be equal theta gaps: the scale is non-linear in score
    theta, _b = rasch_fit(_GUTTMAN)
    gaps = [round(theta[i + 1] - theta[i], 6) for i in range(len(theta) - 1)]
    assert len(set(gaps)) > 1                  # not a constant (i.e. not just a relabeled rank)


def test_point_biserial_high_for_discriminating_item_zero_for_flat():
    # col 0: separates top from bottom (discriminates); col 2: everyone same (flat -> None)
    matrix = [
        [1, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
    r = point_biserial(matrix)
    assert r[0] is not None and r[0] > 0.5     # item 0 tracks total score
    assert r[1] is None                        # flat item (all correct) -> undefined
    assert r[2] is None                        # flat item (all wrong) -> undefined


def test_empty_matrix():
    assert rasch_fit([]) == ([], [])
    assert point_biserial([]) == []
