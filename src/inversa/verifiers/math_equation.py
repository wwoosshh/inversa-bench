"""Forward-verification oracle for posed single-variable equations.

Given a target value, check whether a model-posed equation in `x` is
well-formed and actually has that target among its real solutions.
This is the slice-1 "validity" layer (depth constraints come later).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_X = sp.Symbol("x")
_TOL = 1e-6  # max |solution - target| to count as a match (sympy returns exact/near-exact floats)

# parse_expr has eval-like semantics; restrict the namespace to sympy names with no
# builtins so untrusted model output cannot execute arbitrary Python (e.g. __import__).
_SAFE_GLOBAL = {k: getattr(sp, k) for k in dir(sp) if not k.startswith("_")}
_SAFE_GLOBAL["__builtins__"] = {}


@dataclass
class VerificationResult:
    well_formed: bool        # parses as an equation in x
    valid: bool              # target is among the real solutions
    unique: bool             # the real solution set is exactly {target}
    solutions: List[str]     # string repr of all solutions (evidence)
    error: Optional[str] = None


def _real_solutions(solutions) -> List[float]:
    reals: List[float] = []
    for sol in solutions:
        try:
            val = complex(sol)
        except (TypeError, ValueError):
            continue  # symbolic / non-numeric solution -> skip
        if abs(val.imag) < 1e-9:
            reals.append(val.real)
    return reals


def verify_equation(equation_str: str, target: float) -> VerificationResult:
    s = (equation_str or "").strip()
    if s.count("=") != 1 or any(op in s for op in ("!=", "<=", ">=")):
        return VerificationResult(False, False, False, [],
                                  "equation must contain exactly one '='")
    lhs_str, rhs_str = s.split("=")
    try:
        lhs = parse_expr(lhs_str, transformations=_TRANSFORMS,
                         global_dict=_SAFE_GLOBAL, local_dict={"x": _X}, evaluate=True)
        rhs = parse_expr(rhs_str, transformations=_TRANSFORMS,
                         global_dict=_SAFE_GLOBAL, local_dict={"x": _X}, evaluate=True)
    except Exception as e:  # parse failure -> not well formed
        return VerificationResult(False, False, False, [], f"parse error: {e}")

    if _X not in (lhs - rhs).free_symbols:
        return VerificationResult(False, False, False, [],
                                  "equation does not constrain x (x absent or cancels out)")

    eq = sp.Eq(lhs, rhs)
    try:
        solutions = sp.solve(eq, _X, dict=False)
    except Exception as e:
        return VerificationResult(True, False, False, [], f"solve error: {e}")

    reals = _real_solutions(solutions)
    target_f = float(target)
    valid = any(abs(rs - target_f) < _TOL for rs in reals)
    unique = valid and len(reals) == 1
    return VerificationResult(True, valid, unique, [str(sol) for sol in solutions])
