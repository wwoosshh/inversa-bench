"""Adversarial posing (model-vs-model): the poser must construct a VALID equation
(target is a real solution, confirmed by sympy) that a weaker SOLVER model gets
WRONG. adversarial_success = valid AND weak model solved it incorrectly. This
measures the poser's model-of-difficulty, decoupled from its own solving."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import sympy as sp

from inversa.adapters.base import Adapter
from inversa.tasks.posing import extract_equation
from inversa.verifiers.math_equation import verify_equation, VerificationResult

POSER_PROMPT = (
    "You are constructing a HARD math problem.\n"
    "Write a single-variable algebraic EQUATION in the variable x whose UNIQUE real "
    "solution is exactly {target}, but make it tricky enough that a LESS capable solver "
    "would get it wrong.\n"
    "Output ONLY the equation as '<lhs> = <rhs>', with no explanation. Use sympy syntax."
)

SOLVER_PROMPT = (
    "Solve this equation for x:\n{equation}\n"
    "Output ONLY the numeric value of x (a single number), nothing else."
)

_TOL = 1e-6
_NUM_RE = re.compile(r"-?\d+/\d+|-?\d+\.?\d*")
_XEQ_RE = re.compile(r"x\s*=\s*(-?\d+/\d+|-?\d+\.?\d*)", re.IGNORECASE)
_NOSOL_RE = re.compile(r"\bno\s+(real\s+)?solution", re.IGNORECASE)


def build_poser_prompt(target) -> str:
    return POSER_PROMPT.format(target=target)


def build_solver_prompt(equation: str) -> str:
    return SOLVER_PROMPT.format(equation=equation)


def parse_numeric_answer(text: str) -> Optional[float]:
    """Best-effort extraction of a single numeric answer from a model's solve reply.
    Priority: explicit 'x = <num>' anywhere in the text; else the whole cleaned reply;
    else the FIRST number token. Returns None for 'no solution' replies or when no
    number is present."""
    s = (text or "").strip()
    if _NOSOL_RE.search(s):
        return None
    m = _XEQ_RE.search(s)
    if m:
        try:
            return float(sp.sympify(m.group(1)))
        except Exception:
            pass
    s2 = re.sub(r"^\s*x\s*=\s*", "", s, flags=re.IGNORECASE)
    for cand in [s2] + _NUM_RE.findall(s2):
        try:
            return float(sp.sympify(cand))
        except Exception:
            continue
    return None


@dataclass
class AdversarialItem:
    target: float
    raw_poser_output: str
    equation: str
    verification: VerificationResult
    valid: bool                  # target is a real solution (sympy)
    weak_raw_output: str
    weak_answer: Optional[float]
    weak_correct: bool
    adversarial_success: bool    # valid AND weak model got it wrong


def run_adversarial_item(poser: Adapter, weak_solver: Adapter, target) -> AdversarialItem:
    raw = poser.generate(build_poser_prompt(target))
    equation = extract_equation(raw)
    ver = verify_equation(equation, target)
    weak_raw = weak_solver.generate(build_solver_prompt(equation))
    weak_ans = parse_numeric_answer(weak_raw)
    weak_correct = weak_ans is not None and abs(weak_ans - float(target)) < _TOL
    return AdversarialItem(
        target=float(target),
        raw_poser_output=raw,
        equation=equation,
        verification=ver,
        valid=ver.valid,
        weak_raw_output=weak_raw,
        weak_answer=weak_ans,
        weak_correct=weak_correct,
        adversarial_success=(ver.valid and not weak_correct),
    )


def run_adversarial_batch(poser: Adapter, weak_solver: Adapter, targets) -> list:
    return [run_adversarial_item(poser, weak_solver, t) for t in targets]
