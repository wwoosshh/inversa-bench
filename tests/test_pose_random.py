"""Randomized pose targets close weakness #3 (pose contamination): instead of a FIXED public bank
of targets (memorizable once leaked), targets are generated fresh at test time. The generator must
be (a) correct — each target_desc must symbolically equal its target_value (no answer drift),
(b) deterministic given a seed (auditable/reproducible), and (c) actually random across seeds
(contamination-immune). All checked here without any network.
"""
from __future__ import annotations

import random

import sympy as sp

from inversa.pose_random import random_pose_targets

_T = sp.Symbol("t")


def _desc_matches_value(desc: str, value: float) -> bool:
    """A target is well-formed iff its description really denotes its numeric value."""
    if "real root of" in desc:
        poly = desc.split("real root of", 1)[1].split("=")[0]
        return abs(float(sp.sympify(poly).subs(_T, value))) < 1e-6
    return abs(float(sp.sympify(desc)) - value) < 1e-6


def test_generates_requested_count_with_required_fields():
    ts = random_pose_targets(12, random.Random(0))
    assert len(ts) == 12
    for t in ts:
        assert set(t) >= {"target_desc", "target_value", "novelty"}
        assert isinstance(t["target_value"], float)


def test_every_target_desc_symbolically_equals_its_value():
    # the correctness guard: a generator bug that drifts desc from value is caught here
    for t in random_pose_targets(40, random.Random(7)):
        assert _desc_matches_value(t["target_desc"], t["target_value"]), t


def test_same_seed_is_reproducible():
    a = random_pose_targets(10, random.Random(42))
    b = random_pose_targets(10, random.Random(42))
    assert [x["target_desc"] for x in a] == [x["target_desc"] for x in b]


def test_different_seeds_give_different_targets():
    # contamination-immunity: fresh targets each run -> no fixed item to leak
    a = random_pose_targets(12, random.Random(1))
    b = random_pose_targets(12, random.Random(2))
    assert [x["target_desc"] for x in a] != [x["target_desc"] for x in b]


def test_spans_multiple_novelty_categories():
    cats = {t["novelty"] for t in random_pose_targets(40, random.Random(3))}
    assert len(cats) >= 4  # not all one trivial kind
