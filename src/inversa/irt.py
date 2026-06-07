"""Item Response Theory (Rasch / 1PL) scaling for IGS — closes weakness #6 (IGS is only ordinal).

A raw validity mean over a convenience bank is an ordinal index: a 0.9-vs-0.8 gap is not metrically
meaningful, and pose/transform are averaged with arbitrary equal weight. Fitting a Rasch model to
the model x item correctness matrix instead puts model ability on an interval (logit) theta scale
and assigns each item a difficulty b, with P(correct) = sigmoid(theta_model - b_item). Pooling pose
and transform items into one matrix also drops the arbitrary 50/50 weighting (each item contributes
by its own information). `point_biserial` reports classical item discrimination.

Dependency-free joint MLE (alternating Newton steps), so it composes with the rest of analysis.py
with no scipy/numpy. Perfect/zero responders and all-pass/all-fail items have no finite estimate;
their theta/b are clamped to +/-6 logits (standard JMLE practice) and should be read as boundary.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from inversa.analysis import _pearson

_CLAMP = 6.0  # logit bound for perfect/zero responders & items (no finite JMLE estimate)


def _sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def rasch_fit(matrix: Sequence[Sequence[float]], iters: int = 300,
              tol: float = 1e-7) -> Tuple[List[float], List[float]]:
    """Fit a Rasch (1PL) model to a J-models x I-items 0/1 matrix. Returns (thetas, difficulties):
    thetas[j] = model j ability (logits), b[i] = item i difficulty (logits, centered at 0 for
    identifiability). theta is a monotone but NON-linear function of raw score, i.e. an interval
    scale, not just a rank relabeling."""
    if not matrix or not matrix[0]:
        return ([], [])
    J, I = len(matrix), len(matrix[0])
    x = [[float(matrix[j][i]) for i in range(I)] for j in range(J)]
    theta = [0.0] * J
    b = [0.0] * I

    def clamp(v: float) -> float:
        return max(-_CLAMP, min(_CLAMP, v))

    for _ in range(iters):
        moved = 0.0
        for j in range(J):  # Newton step on theta_j holding b
            g = h = 0.0
            for i in range(I):
                p = _sigmoid(theta[j] - b[i])
                g += x[j][i] - p
                h += p * (1.0 - p)
            if h > 1e-12:
                nt = clamp(theta[j] + g / h)
                moved = max(moved, abs(nt - theta[j]))
                theta[j] = nt
        for i in range(I):  # Newton step on b_i holding theta
            g = h = 0.0
            for j in range(J):
                p = _sigmoid(theta[j] - b[i])
                g += p - x[j][i]
                h += p * (1.0 - p)
            if h > 1e-12:
                nb = clamp(b[i] + g / h)
                moved = max(moved, abs(nb - b[i]))
                b[i] = nb
        mb = sum(b) / I  # center difficulties (P depends only on theta-b -> shift both by mean(b))
        b = [bi - mb for bi in b]
        theta = [t - mb for t in theta]
        if moved < tol:
            break
    return theta, b


def point_biserial(matrix: Sequence[Sequence[float]]) -> List[Optional[float]]:
    """Classical item discrimination: Pearson correlation of each item's 0/1 column with the
    per-model total score. High = the item separates strong from weak models; None for a flat
    item (all-pass / all-fail) where discrimination is undefined."""
    if not matrix or not matrix[0]:
        return []
    J, I = len(matrix), len(matrix[0])
    totals = [sum(matrix[j]) for j in range(J)]
    out: List[Optional[float]] = []
    for i in range(I):
        col = [float(matrix[j][i]) for j in range(J)]
        out.append(_pearson(col, totals) if len(set(col)) > 1 else None)
    return out
