"""E8: harder transform bank — POLYNOMIAL transforms (root = r**2, r**3, r**2-2) instead of
Mobius. These require eliminating/raising powers (resultant-style reasoning), not a single
substitution, so they should resolve the IGS top (currently 3 models tied at 1.0). Uniqueness is
preserved (source has 1 real root → its image under a real polynomial has a degree-3 minimal
polynomial with exactly 1 real root, verified). Run: python scripts/gen_transform_hard_bank.py
"""
import json

import sympy as sp

x = sp.Symbol("x")
CUBICS = [(1, -1), (1, -3), (2, -5), (-1, -1), (3, -2), (1, 1)]
TR = [("r**2", lambda a: a**2), ("r**3", lambda a: a**3), ("r**2 - 2", lambda a: a**2 - 2)]

items = []
for p, q in CUBICS:
    src = x**3 + p * x + q
    src_str = f"{sp.printing.sstr(src)} = 0"
    a = [r for r in sp.Poly(src, x).all_roots() if r.is_real][0]
    for desc, g in TR:
        gv = g(a)
        mp = sp.minimal_polynomial(gv, x)
        nreal = sum(1 for r in sp.Poly(mp, x).all_roots() if r.is_real)
        items.append({"source": src_str, "g_desc": desc, "r_value": float(a),
                      "target_value": float(gv), "canonical_solvable": nreal == 1})

json.dump({"name": "transform_hard",
           "note": "Polynomial root transforms (r^2, r^3, r^2-2) — require power-raising/elimination, "
                   "harder than Mobius; uniqueness verified via minimal polynomial (1 real root).",
           "items": items},
          open("data/banks/transform_hard.json", "w", encoding="utf-8"), indent=2)
print(f"wrote {len(items)} hard transform items; all solvable:",
      all(i["canonical_solvable"] for i in items))
