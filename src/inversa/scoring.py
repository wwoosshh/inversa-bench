"""Inversa Generative Score (IGS) — a standalone, recall-proof indicator of a model's ability
to CONSTRUCT mathematical structure. Designed to discriminate models in the regime where
forward-SOLVING benchmarks have saturated (many models at 100%, so the top is unmeasurable).

IGS combines two sympy-verified, memorization-resistant axes (each in [0,1]):
  - pose validity: construct an equation whose UNIQUE real solution is a given (often
    irrational/structural) target. The naive template — a minimal polynomial — carries
    conjugate roots and fails uniqueness, so success needs genuine construction.
  - transform validity: structurally transform a FRESHLY-RANDOMIZED source equation so its
    unique real root becomes g(r). The answer is pinned to a test-time input, so a memorized
    problem cannot satisfy it.

IGS = mean(pose_validity, transform_validity). Higher = stronger constructor. It is generative-
ONLY (solving is never part of it); solve is reported separately as a saturation reference.
"""
from __future__ import annotations

from dataclasses import dataclass

from inversa.tasks.structural import validity_rate


def igs(pose_validity: float, transform_validity: float) -> float:
    """The composite generative score: equal-weight mean of the two recall-proof axes."""
    return round((pose_validity + transform_validity) / 2.0, 4)


@dataclass
class GenerativeScore:
    model: str
    pose_validity: float
    transform_validity: float
    igs: float
    n_pose: int = 0
    n_transform: int = 0


def generative_score(model: str, pose_items, transform_items) -> GenerativeScore:
    pv = validity_rate(pose_items)
    tv = validity_rate(transform_items)
    return GenerativeScore(model, pv, tv, igs(pv, tv), len(pose_items), len(transform_items))


def score_from_rates(model: str, pose_validity: float, transform_validity: float) -> GenerativeScore:
    """Build an IGS from already-aggregated rates (e.g. loaded from a results JSON)."""
    return GenerativeScore(model, pose_validity, transform_validity, igs(pose_validity, transform_validity))
