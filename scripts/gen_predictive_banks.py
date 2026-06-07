"""Generate the E14 predictive-validity banks, with keys verified by sympy (correct by construction).

Y1 (construct_constraints.json): constrained polynomial construction. Each item = a constraint list;
the prompt is templated FROM the constraints (so prose can't drift from what's graded), and a reference
solution is asserted to satisfy `check_construction` at build time (so every item is satisfiable).

Y2 (verify_pairs.json): error detection. Each yes/no claim's truth is COMPUTED by sympy here, so the
answer key is correct by construction; the model is later scored by exact yes/no match.
"""
from __future__ import annotations

import json

import sympy as sp

from inversa.predictive import check_construction

X = sp.Symbol("x")

# ---- Y1: (checks, reference solution that must satisfy them) --------------------------------
Y1 = [
    ([{"type": "degree", "value": 2}, {"type": "integer_coeffs"}, {"type": "root_at", "x": 3},
      {"type": "leading_coeff", "value": 1}], "x**2 - 3*x = 0"),
    ([{"type": "degree", "value": 2}, {"type": "integer_coeffs"}, {"type": "root_at", "x": -2},
      {"type": "leading_coeff", "value": 1}], "x**2 + 2*x = 0"),
    ([{"type": "degree", "value": 2}, {"type": "integer_coeffs"}, {"type": "root_at", "x": 5},
      {"type": "root_at", "x": -5}, {"type": "leading_coeff", "value": 1}], "x**2 - 25 = 0"),
    ([{"type": "degree", "value": 2}, {"type": "integer_coeffs"},
      {"type": "num_distinct_real_roots", "value": 0}, {"type": "leading_coeff", "value": 1}],
     "x**2 + 1 = 0"),
    ([{"type": "degree", "value": 3}, {"type": "integer_coeffs"}, {"type": "root_at", "x": 0},
      {"type": "leading_coeff", "value": 1}, {"type": "num_distinct_real_roots", "value": 3}],
     "x**3 - x = 0"),
    ([{"type": "degree", "value": 3}, {"type": "integer_coeffs"}, {"type": "root_at", "x": 2},
      {"type": "leading_coeff", "value": 1}, {"type": "num_distinct_real_roots", "value": 1}],
     "x**3 - 2*x**2 + x - 2 = 0"),
    ([{"type": "degree", "value": 3}, {"type": "integer_coeffs"}, {"type": "leading_coeff", "value": 2},
      {"type": "root_at", "x": 1}], "2*x**3 - 2 = 0"),
    ([{"type": "degree", "value": 4}, {"type": "integer_coeffs"}, {"type": "root_at", "x": 1},
      {"type": "leading_coeff", "value": 1}, {"type": "num_distinct_real_roots", "value": 2}],
     "x**4 - 1 = 0"),
    ([{"type": "degree", "value": 4}, {"type": "integer_coeffs"}, {"type": "leading_coeff", "value": 1},
      {"type": "num_distinct_real_roots", "value": 0}], "x**4 + 1 = 0"),
    ([{"type": "degree", "value": 4}, {"type": "integer_coeffs"}, {"type": "root_at", "x": 3},
      {"type": "leading_coeff", "value": 1}, {"type": "num_distinct_real_roots", "value": 2}],
     "x**4 - 8*x**2 - 9 = 0"),
    ([{"type": "degree", "value": 4}, {"type": "integer_coeffs"}, {"type": "root_at", "x": 2},
      {"type": "leading_coeff", "value": 1}, {"type": "num_distinct_real_roots", "value": 1}],
     "(x-2)**2*(x**2+1) = 0"),
    ([{"type": "degree", "value": 5}, {"type": "integer_coeffs"}, {"type": "leading_coeff", "value": 1},
      {"type": "num_distinct_real_roots", "value": 1}], "x**5 + x = 0"),
    ([{"type": "unique_real_root", "value": 2.0}, {"type": "integer_coeffs"},
      {"type": "degree", "value": 3}], "x**3 - 2*x**2 + x - 2 = 0"),
    ([{"type": "unique_real_root", "value": -1.0}, {"type": "integer_coeffs"},
      {"type": "degree", "value": 3}], "x**3 + x**2 + x + 1 = 0"),
    ([{"type": "degree", "value": 3}, {"type": "integer_coeffs"}, {"type": "root_at", "x": 4},
      {"type": "root_at", "x": -1}, {"type": "leading_coeff", "value": 1},
      {"type": "num_distinct_real_roots", "value": 3}], "(x-4)*(x+1)*(x-2) = 0"),
    ([{"type": "unique_real_root", "value": 3.0}, {"type": "integer_coeffs"},
      {"type": "degree", "value": 5}], "(x-3)*(x**2+1)*(x**2+x+1) = 0"),
]

_PHRASE = {
    "degree": lambda c: f"degree exactly {c['value']}",
    "integer_coeffs": lambda c: "integer coefficients",
    "root_at": lambda c: f"a root at x = {c['x']} (so p({c['x']}) = 0)",
    "leading_coeff": lambda c: f"leading coefficient {c['value']}",
    "num_distinct_real_roots": lambda c: f"exactly {c['value']} distinct real root(s)",
    "unique_real_root": lambda c: f"a UNIQUE real root equal to {c['value']:g}",
}


def _y1_prompt(checks):
    parts = "; ".join(_PHRASE[c["type"]](c) for c in checks)
    return ("Construct a single-variable polynomial p(x) with ALL of these properties: " + parts + ". "
            "Output ONLY the polynomial as an equation on a final line prefixed by '#### ', "
            "in the form '#### p(x) = 0'. Example: '#### x**2 - 5*x + 6 = 0'.")


# ---- Y2: yes/no claims; truth computed by sympy -------------------------------------------
def _is_solution(eq_expr, r):
    return abs(float(eq_expr.subs(X, r))) < 1e-9


def _distinct_real(eq_expr):
    return len(set(sp.Poly(sp.expand(eq_expr), X).real_roots()))


# (claim_text, truth) — truth recomputed by sympy below so the key cannot be wrong
_Y2_SPEC = [
    ("Is x = 3 a real solution of x**3 - x**2 - 9*x + 9 = 0?", lambda: _is_solution(X**3 - X**2 - 9*X + 9, 3)),
    ("Is x = 2 a real solution of x**3 - x**2 - 9*x + 9 = 0?", lambda: _is_solution(X**3 - X**2 - 9*X + 9, 2)),
    ("Is x = 2 a real solution of x**2 - 5*x + 6 = 0?", lambda: _is_solution(X**2 - 5*X + 6, 2)),
    ("Is x = 4 a real solution of x**2 - 5*x + 6 = 0?", lambda: _is_solution(X**2 - 5*X + 6, 4)),
    ("Is x = 3 the UNIQUE real solution of x**3 - x**2 - 9*x + 9 = 0?", lambda: set(sp.Poly(X**3 - X**2 - 9*X + 9, X).real_roots()) == {sp.Integer(3)}),
    ("Is x = 1 the UNIQUE real solution of x**3 + x - 2 = 0?", lambda: set(sp.Poly(X**3 + X - 2, X).real_roots()) == {sp.Integer(1)}),
    ("Does x**2 - 4 = 0 have exactly 2 distinct real solutions?", lambda: _distinct_real(X**2 - 4) == 2),
    ("Does x**2 + 4 = 0 have exactly 2 distinct real solutions?", lambda: _distinct_real(X**2 + 4) == 2),
    ("Does x**3 - x = 0 have exactly 3 distinct real solutions?", lambda: _distinct_real(X**3 - X) == 3),
    ("Does x**4 - 1 = 0 have exactly 4 distinct real solutions?", lambda: _distinct_real(X**4 - 1) == 4),
    ("Is (x - 2)*(x - 3) a correct factorization of x**2 - 5*x + 6?", lambda: sp.expand((X-2)*(X-3)) == sp.expand(X**2 - 5*X + 6)),
    ("Is (x - 1)*(x - 6) a correct factorization of x**2 - 5*x + 6?", lambda: sp.expand((X-1)*(X-6)) == sp.expand(X**2 - 5*X + 6)),
    ("Is (x + 1)*(x**2 + 1) a correct factorization of x**3 + x**2 + x + 1?", lambda: sp.expand((X+1)*(X**2+1)) == sp.expand(X**3 + X**2 + X + 1)),
    ("Is x = -3 a real solution of x**3 - x**2 - 9*x + 9 = 0?", lambda: _is_solution(X**3 - X**2 - 9*X + 9, -3)),
    ("Is x = 0 a real solution of x**3 - x = 0?", lambda: _is_solution(X**3 - X, 0)),
    ("Is x = 2 the UNIQUE real solution of x**3 - 2*x**2 + x - 2 = 0?", lambda: set(sp.Poly(X**3 - 2*X**2 + X - 2, X).real_roots()) == {sp.Integer(2)}),
    ("Does x**4 + 1 = 0 have exactly 0 distinct real solutions?", lambda: _distinct_real(X**4 + 1) == 0),
    ("Does (x - 1)**2*(x + 1) = 0 have exactly 3 distinct real solutions?", lambda: _distinct_real((X-1)**2*(X+1)) == 3),
    ("Is x = 5 a real solution of x**2 - 25 = 0?", lambda: _is_solution(X**2 - 25, 5)),
    ("Is x = 5 a real solution of x**2 + 25 = 0?", lambda: _is_solution(X**2 + 25, 5)),
    ("Is (x - 2)*(x + 2)*(x**2 + 1) a correct factorization of x**4 - 16?", lambda: sp.expand((X-2)*(X+2)*(X**2+1)) == sp.expand(X**4 - 16)),
    ("Is x = 1 the UNIQUE real solution of x**5 + x - 2 = 0?", lambda: set(sp.Poly(X**5 + X - 2, X).real_roots()) == {sp.Integer(1)}),
    ("Does x**2 - 6*x + 9 = 0 have exactly 2 distinct real solutions?", lambda: _distinct_real(X**2 - 6*X + 9) == 2),
    ("Does x**2 - 6*x + 9 = 0 have exactly 1 distinct real solution?", lambda: _distinct_real(X**2 - 6*X + 9) == 1),
]


def main():
    y1 = []
    for i, (checks, ref) in enumerate(Y1, 1):
        ok, _ = check_construction(checks, ref)
        assert ok, f"Y1 item {i}: reference {ref!r} does not satisfy its own checks"
        y1.append({"id": i, "prompt": _y1_prompt(checks), "checks": checks})
    json.dump({"name": "construct-constraints-v1",
               "note": "E14 Y1: constrained polynomial construction; sympy-graded, references verified.",
               "items": y1},
              open("data/banks/construct_constraints.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    y2 = []
    for i, (claim, truth_fn) in enumerate(_Y2_SPEC, 1):
        truth = bool(truth_fn())
        y2.append({"id": i, "prompt": claim + " Answer with ONLY 'yes' or 'no' on a final line "
                                            "prefixed by '#### '. Example: '#### no'.",
                   "answer": "yes" if truth else "no"})
    n_yes = sum(1 for q in y2 if q["answer"] == "yes")
    json.dump({"name": "verify-pairs-v1",
               "note": f"E14 Y2: yes/no error-detection claims; truth sympy-computed ({n_yes} yes / "
                       f"{len(y2)-n_yes} no).", "items": y2},
              open("data/banks/verify_pairs.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"Y1: {len(y1)} construction items (all references verified)")
    print(f"Y2: {len(y2)} verify claims ({n_yes} yes / {len(y2)-n_yes} no)")


if __name__ == "__main__":
    main()
