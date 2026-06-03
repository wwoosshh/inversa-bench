"""Inverse/posing task: target answer -> model constructs an equation -> verify."""
from __future__ import annotations

from dataclasses import dataclass

from inversa.adapters.base import Adapter
from inversa.verifiers.math_equation import verify_equation, VerificationResult

PROMPT_TEMPLATE = (
    "You are constructing a math problem.\n"
    "Write a single-variable algebraic EQUATION in the variable x whose unique "
    "solution is exactly {target}.\n"
    "Output ONLY the equation in the form '<lhs> = <rhs>', with no explanation.\n"
    "Use Python/sympy syntax (examples: 2*x + 1 = 7 ; x**2 - 9 = 0).\n"
)


def build_prompt(target) -> str:
    return PROMPT_TEMPLATE.format(target=target)


def _extract_equation(raw: str) -> str:
    for line in raw.splitlines():
        if "=" in line:
            return line.strip().strip("`").strip()
    return raw.strip().strip("`").strip()


@dataclass
class ItemResult:
    target: float
    raw_output: str
    equation: str
    verification: VerificationResult


def run_item(adapter: Adapter, target) -> ItemResult:
    raw = adapter.generate(build_prompt(target))
    equation = _extract_equation(raw)
    verification = verify_equation(equation, target)
    return ItemResult(float(target), raw, equation, verification)


def run_batch(adapter: Adapter, targets) -> list:
    return [run_item(adapter, t) for t in targets]
