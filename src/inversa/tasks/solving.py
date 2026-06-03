"""Forward solving eval: ask a model to solve fixed equations; theta_solve = accuracy.
Reuses the slice-2b solver prompt + numeric-answer parser."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from inversa.adapters.base import Adapter
from inversa.tasks.adversarial import build_solver_prompt, parse_numeric_answer

_TOL = 1e-6


@dataclass
class SolveItem:
    equation: str
    expected: float
    raw_output: str
    answer: Optional[float]
    correct: bool


def solve_item(adapter: Adapter, equation: str, expected) -> SolveItem:
    raw = adapter.generate(build_solver_prompt(equation))
    ans = parse_numeric_answer(raw)
    correct = ans is not None and abs(ans - float(expected)) < _TOL
    return SolveItem(equation, float(expected), raw, ans, correct)


def solve_batch(adapter: Adapter, problems) -> list:
    return [solve_item(adapter, p["equation"], p["answer"]) for p in problems]


def accuracy(items) -> float:
    n = len(items)
    return (sum(1 for it in items if it.correct) / n) if n else 0.0
