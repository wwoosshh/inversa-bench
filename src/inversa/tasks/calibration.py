"""Difficulty-targeted posing: ask the model to pose at a requested degree and
measure validity + calibration (|requested - actual degree band|)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from inversa.adapters.base import Adapter
from inversa.tasks.posing import _extract_equation
from inversa.verifiers.difficulty import difficulty, DifficultyResult
from inversa.verifiers.math_equation import verify_equation, VerificationResult

CALIB_PROMPT = (
    "You are constructing a math problem.\n"
    "Write a single-variable algebraic EQUATION in the variable x whose unique "
    "solution is exactly {target} AND whose polynomial degree in x is exactly {level}.\n"
    "Output ONLY the equation as '<lhs> = <rhs>', with no explanation. "
    "Use sympy syntax (e.g. x**2 - 4 = 0 has degree 2)."
)


def build_calibration_prompt(target, level: int) -> str:
    return CALIB_PROMPT.format(target=target, level=level)


@dataclass
class CalibrationItem:
    target: float
    requested_level: int
    raw_output: str
    equation: str
    verification: VerificationResult
    difficulty: DifficultyResult
    valid: bool                       # solves to target
    calibration_error: Optional[int]  # |requested_level - actual band|, None if band==0


def run_calibration_item(adapter: Adapter, target, level: int) -> CalibrationItem:
    raw = adapter.generate(build_calibration_prompt(target, level))
    equation = _extract_equation(raw)
    ver = verify_equation(equation, target)
    dif = difficulty(equation)
    cal_err = abs(int(level) - dif.band) if dif.band else None
    return CalibrationItem(
        target=float(target),
        requested_level=int(level),
        raw_output=raw,
        equation=equation,
        verification=ver,
        difficulty=dif,
        valid=ver.valid,
        calibration_error=cal_err,
    )


def run_calibration_batch(adapter: Adapter, targets, levels) -> list:
    return [run_calibration_item(adapter, t, lv) for t in targets for lv in levels]
