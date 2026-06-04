"""Forward-verification oracle for posed single-variable equations.

Given a target value, check whether a model-posed equation in `x` is
well-formed and actually has that target among its real solutions.
This is the slice-1 "validity" layer (depth constraints come later).
"""
from __future__ import annotations

import threading
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

_SOLVE_TIMEOUT = 10.0  # seconds; sp.solve can spin forever on pathological transcendental input


def _solve_with_timeout(eq, sym, timeout: float = _SOLVE_TIMEOUT):
    """Run sp.solve on a daemon thread and abandon it past `timeout`. sympy has no native
    timeout and can hang indefinitely on adversarial model output; this bounds every verify.
    A timed-out solve raises TimeoutError (caller treats it as an unverifiable -> invalid pose)."""
    box: dict = {}

    def _work():
        try:
            box["sol"] = sp.solve(eq, sym, dict=False)
        except Exception as exc:  # noqa: BLE001 - re-raised on the calling thread
            box["err"] = exc

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"solve exceeded {timeout}s")
    if "err" in box:
        raise box["err"]
    return box["sol"]


def _eval_real(f, val) -> Optional[float]:
    """Evaluate a lambdified function at a real point; return the real value, or None if
    it is undefined there (outside domain), non-real, NaN or infinite."""
    try:
        y = f(val)
        yc = complex(y)
    except Exception:
        return None
    if abs(yc.imag) > 1e-7:
        return None
    r = yc.real
    if r != r or abs(r) == float("inf"):
        return None
    return r


def _bisect(f, a, b, iters: int = 60):
    fa, fb = _eval_real(f, a), _eval_real(f, b)
    if fa is None or fb is None or (fa < 0) == (fb < 0):
        return None
    for _ in range(iters):
        m = (a + b) / 2.0
        fm = _eval_real(f, m)
        if fm is None:
            return None
        if abs(fm) < 1e-12:
            return m
        if (fa < 0) != (fm < 0):
            b = m
        else:
            a, fa = m, fm
    return (a + b) / 2.0


def _scan_real_roots(f, target: float, half: float = 50.0, step: float = 0.1) -> List[float]:
    """Heuristic: real roots of f within [target-half, target+half] via sign-change scan +
    bisection. Bounded window -> may miss far-away roots; good enough to corroborate the
    uniqueness of transcendental constructions our symbolic solver cannot handle."""
    lo = target - half
    n = int((2 * half) / step)
    roots: List[float] = []
    prev_x = prev_y = None
    for i in range(n + 1):
        x = lo + i * step
        y = _eval_real(f, x)
        if y is not None and abs(y) < 1e-9:
            roots.append(x)
        elif prev_y is not None and y is not None and (prev_y < 0) != (y < 0):
            r = _bisect(f, prev_x, x)
            if r is not None:
                roots.append(r)
        prev_x, prev_y = x, y
    uniq: List[float] = []
    for r in sorted(roots):
        if not uniq or abs(r - uniq[-1]) > 1e-6:
            uniq.append(r)
    return uniq


def _numeric_verify(expr, target: float):
    """Fallback when symbolic solve can't confirm the target (e.g. transcendental embeddings
    like exp(g(x))=1). Returns (valid, unique, roots) or None if expr can't be evaluated.
    Membership is exact; uniqueness is the heuristic root-count within a bounded window."""
    f = sp.lambdify(_X, expr, modules=["mpmath"])
    yt = _eval_real(f, target)
    if yt is None:
        return None
    if abs(yt) >= 1e-6:
        return (False, False, [])  # target is not a root; skip the expensive scan
    roots = _scan_real_roots(f, target)
    if not any(abs(r - target) < 1e-4 for r in roots):
        roots = sorted(roots + [target])
    return (True, len(roots) == 1, roots)


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


def parse_sides(equation_str: str):
    """Parse 'lhs = rhs' into (lhs, rhs) sympy expressions using the sandboxed
    namespace. Raises ValueError if the string is not a single well-formed
    equation (wrong count of '=', inequality operators, or parse failure)."""
    s = (equation_str or "").strip()
    if s.count("=") != 1 or any(op in s for op in ("!=", "<=", ">=", "<", ">")):
        raise ValueError("equation must contain exactly one '='")
    lhs_str, rhs_str = s.split("=")
    try:
        lhs = parse_expr(lhs_str, transformations=_TRANSFORMS,
                         local_dict={"x": _X}, global_dict=_SAFE_GLOBAL, evaluate=True)
        rhs = parse_expr(rhs_str, transformations=_TRANSFORMS,
                         local_dict={"x": _X}, global_dict=_SAFE_GLOBAL, evaluate=True)
    except Exception as e:
        raise ValueError(f"parse error: {e}")
    # A side must be a plain arithmetic expression. Models sometimes emit sympy relational
    # syntax (e.g. "Eq(x**2, 4) = 0"), which parses to a Relational/Boolean — not an Expr —
    # and would crash later arithmetic (Equality - Zero). Reject it as malformed.
    if not isinstance(lhs, sp.Expr) or not isinstance(rhs, sp.Expr):
        raise ValueError("each side must be a plain arithmetic expression (got a relational/boolean)")
    return lhs, rhs


def verify_equation(equation_str: str, target: float) -> VerificationResult:
    try:
        lhs, rhs = parse_sides(equation_str)
    except ValueError as e:
        return VerificationResult(False, False, False, [], str(e))

    if _X not in (lhs - rhs).free_symbols:
        return VerificationResult(False, False, False, [],
                                  "equation does not constrain x (x absent or cancels out)")

    eq = sp.Eq(lhs, rhs)
    target_f = float(target)
    try:
        solutions = _solve_with_timeout(eq, _X)
    except Exception:
        solutions = None

    if solutions is not None:
        reals = _real_solutions(solutions)
        if any(abs(rs - target_f) < _TOL for rs in reals):
            # symbolic solve is authoritative when it confirms the target as a root
            return VerificationResult(True, True, len(reals) == 1, [str(s) for s in solutions])

    # Symbolic couldn't confirm the target (transcendental embedding, timeout, or genuinely
    # absent). Fall back to numeric verification so valid-but-hard-to-symbolically-solve
    # constructions are not falsely rejected.
    num = _numeric_verify(lhs - rhs, target_f)
    if num is not None:
        valid, unique, roots = num
        note = None if valid else "target is not a real root (numeric)"
        return VerificationResult(True, valid, unique, [f"~{r:.6f}" for r in roots], note)

    sols_repr = [str(s) for s in solutions] if solutions is not None else []
    return VerificationResult(True, False, False, sols_repr, "unverifiable (symbolic + numeric)")
