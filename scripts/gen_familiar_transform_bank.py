"""E12b data prep: a transform bank whose SOURCES are famous/textbook cubics (cube roots,
the plastic number, Newton's x^3-2x-5) — all with a single IRRATIONAL real root so difficulty
matches the random-source bank. If transform validity is the same on familiar vs random sources,
the inverse task gains no advantage from familiarity → it is contamination-immune (the structural
claim, confirmed numerically). Run: python scripts/gen_familiar_transform_bank.py
"""
import json

import sympy as sp

from inversa.verifiers.math_equation import verify_equation

X = sp.Symbol("x")
# famous cubics x^3 + p x + q with one real (irrational) root
FAMILIAR = [(0, -2), (0, -3), (-1, -1), (1, -1), (-2, -5), (3, -1)]
TRANSFORMS = [("r + 1", X - 1, lambda r: r + 1),
              ("2*r", X / 2, lambda r: 2 * r),
              ("1/r", 1 / X, lambda r: 1 / r),
              ("1/(r + 1)", (1 - X) / X, lambda r: 1 / (r + 1)),
              ("r/(r + 1)", X / (1 - X), lambda r: r / (r + 1))]


def real_root(expr):
    reals = [c.real for c in (complex(s) for s in sp.solve(sp.Eq(expr, 0), X)) if abs(c.imag) < 1e-9]
    assert len(reals) == 1
    return reals[0]


items = []
for (p, q) in FAMILIAR:
    src = X**3 + p * X + q
    src_str = f"{sp.printing.sstr(src)} = 0"
    r = real_root(src)
    for (g_desc, r_of_x, g_fn) in TRANSFORMS:
        target = float(g_fn(sp.Float(r, 20)))
        num, _ = sp.fraction(sp.together(src.subs(X, r_of_x)))
        solvable = verify_equation(f"{sp.printing.sstr(sp.expand(num))} = 0", target).unique
        items.append({"source": src_str, "g_desc": g_desc, "r_value": float(r),
                      "target_value": target, "canonical_solvable": solvable})

json.dump({"name": "transform_bank_familiar",
           "note": "Famous/textbook cubic sources (irrational roots, difficulty-matched to random) "
                   "for the E12b inverse contamination-gap test.",
           "items": items},
          open("data/banks/transform_bank_familiar.json", "w", encoding="utf-8"), indent=2)
print(f"wrote {len(items)} familiar-source transform items; all solvable:",
      all(i["canonical_solvable"] for i in items))
