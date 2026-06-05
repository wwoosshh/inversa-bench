"""E8+: brutal transform bank — degree-raising / composite polynomial transforms
(root = r**4, r**2+r, r**3-r, r**4-2r). Any real-polynomial image of the single real root keeps a
degree-3 minimal polynomial with one real root (the two complex-conjugate roots map to a conjugate
pair), so uniqueness is preserved & verified. These need deeper elimination than r**2/r**3 —
intended to separate the very top (opus vs qwen3.7, both 100% on the hard bank).
Run: python scripts/gen_transform_brutal_bank.py
"""
import json

import sympy as sp

x = sp.Symbol("x")
CUBICS = [(1, -1), (1, -3), (2, -5), (-1, -1), (3, -2), (1, 1)]
TR = [("r**4", lambda a: a**4), ("r**2 + r", lambda a: a**2 + a),
      ("r**3 - r", lambda a: a**3 - a), ("r**4 - 2*r", lambda a: a**4 - 2 * a)]

items, dropped = [], 0
for p, q in CUBICS:
    src = x**3 + p * x + q
    src_str = f"{sp.printing.sstr(src)} = 0"
    a = [r for r in sp.Poly(src, x).all_roots() if r.is_real][0]
    for desc, g in TR:
        gv = g(a)
        mp = sp.minimal_polynomial(gv, x)
        nreal = sum(1 for r in sp.Poly(mp, x).all_roots() if r.is_real)
        if nreal == 1:
            items.append({"source": src_str, "g_desc": desc, "r_value": float(a),
                          "target_value": float(gv), "canonical_solvable": True})
        else:
            dropped += 1

json.dump({"name": "transform_brutal",
           "note": "Degree-raising/composite polynomial root transforms (r^4, r^2+r, r^3-r, r^4-2r); "
                   "uniqueness verified via minimal polynomial. Hardest construction tier.",
           "items": items},
          open("data/banks/transform_brutal.json", "w", encoding="utf-8"), indent=2)
print(f"wrote {len(items)} brutal transform items ({dropped} dropped for non-uniqueness)")
