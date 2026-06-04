"""Inverse/posing task: target answer -> model constructs an equation -> verify."""
from __future__ import annotations

import re
from dataclasses import dataclass

from inversa.adapters.base import Adapter
from inversa.verifiers.math_equation import verify_equation, VerificationResult

_MARKER_RE = re.compile(r"####\s*(.+)")

PROMPT_TEMPLATE = (
    "You are constructing a math problem.\n"
    "Write a single-variable algebraic EQUATION in the variable x whose unique "
    "solution is exactly {target}.\n"
    "Output ONLY the equation in the form '<lhs> = <rhs>', with no explanation.\n"
    "Use Python/sympy syntax (examples: 2*x + 1 = 7 ; x**2 - 9 = 0).\n"
)


def build_prompt(target) -> str:
    return PROMPT_TEMPLATE.format(target=target)


def extract_equation(raw: str) -> str:
    """Pull the final equation from a model reply. Prefers an explicit '#### <eq>' marker
    (last one wins) so a verbose/reasoning model's final answer is captured; otherwise takes
    the LAST line containing '=' (the conclusion of any working), stripping code fences."""
    s = (raw or "").strip()
    marked = _MARKER_RE.findall(s)
    for cand in reversed(marked):
        c = cand.strip().strip("`").strip()
        if "=" in c:
            return c
    eq_lines = [ln.strip().strip("`").strip() for ln in s.splitlines() if "=" in ln]
    if eq_lines:
        return eq_lines[-1]
    return s.strip("`").strip()


@dataclass
class ItemResult:
    target: float
    raw_output: str
    equation: str
    verification: VerificationResult


def run_item(adapter: Adapter, target) -> ItemResult:
    raw = adapter.generate(build_prompt(target))
    equation = extract_equation(raw)
    verification = verify_equation(equation, target)
    return ItemResult(float(target), raw, equation, verification)


def run_batch(adapter: Adapter, targets) -> list:
    return [run_item(adapter, t) for t in targets]
