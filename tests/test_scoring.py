"""Inversa Generative Score (IGS) definition: equal-weight mean of pose + transform validity."""
from types import SimpleNamespace

from inversa.scoring import generative_score, igs, score_from_rates


def _items(flags):
    return [SimpleNamespace(valid=f) for f in flags]


def test_igs_is_mean_of_axes():
    assert igs(0.5, 1.0) == 0.75
    assert igs(0.0, 0.0) == 0.0
    assert igs(1.0, 1.0) == 1.0


def test_generative_score_from_items():
    pose = _items([True, False, True, False])      # 0.5
    transform = _items([True, True, True, True])    # 1.0
    s = generative_score("m", pose, transform)
    assert s.pose_validity == 0.5
    assert s.transform_validity == 1.0
    assert s.igs == 0.75
    assert s.n_pose == 4 and s.n_transform == 4


def test_score_from_rates_matches():
    s = score_from_rates("m", 0.83, 0.5)
    assert s.igs == igs(0.83, 0.5)


def test_igs_is_generative_only_and_ordered():
    # a model that solves perfectly but constructs poorly scores low on IGS
    weak_constructor = score_from_rates("solver", 0.0, 0.5)   # igs 0.25
    strong_constructor = score_from_rates("builder", 1.0, 1.0)  # igs 1.0
    assert strong_constructor.igs > weak_constructor.igs
