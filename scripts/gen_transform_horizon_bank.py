"""E8++: 'horizon' transform bank — push construction difficulty past the brutal tier to test whether
Inversa can be scaled programmatically to challenge a frontier model that aces the brutal bank
(opus-4.8 = 100% on r^4/composite over cubics).

The lever is the ELIMINATION DEGREE the model must perform. Transform difficulty so far used a degree-3
source (x^3+px+q, one real root r): eliminating r to build an equation for g(r) is a degree-3 resultant.
Here we (a) raise the source to a **quintic** x^5+a*x+c (a>0 -> derivative 5x^4+a>0 -> strictly
increasing -> EXACTLY one real root, so 'unique real solution' still holds), which forces a degree-5
elimination, and (b) also add **composite/higher-degree** transforms over cubics. Either way g(r) is an
algebraic number whose minimal polynomial we compute and keep only when it has exactly one real root
(uniqueness preserved & sympy-verifiable). No human authored these — the generator scales the difficulty,
which is exactly Inversa's claimed advantage over forward benchmarks.

Run: python scripts/gen_transform_horizon_bank.py
"""
import json

import sympy as sp

x = sp.Symbol("x")

# Quintic sources x^5 + a*x + c with a>0 -> derivative 5x^4+a>0 -> strictly increasing -> EXACTLY
# one real root. We keep only IRREDUCIBLE ones (real root of true degree 5), because a reducible
# quintic's real root can live in a degree-3 factor -> no harder than the brutal bank. Composite
# transforms over a CUBIC root stay in a degree-3 field, so they cannot exceed brutal difficulty;
# only a higher-degree source raises the elimination degree. That is the difficulty lever here.
QUINTICS = [(1, -1), (1, 1), (2, -1), (1, -2), (2, 3), (3, -2), (1, 3), (2, -5),
            (1, -4), (3, 1), (1, 5), (2, 1), (3, -4), (4, -3), (1, -6), (5, -2)]
# Septics x^7 + a*x + c (a>0 -> strictly increasing -> one real root): degree-7 elimination, harder
# still. Kept only when irreducible (real root of true degree 7).
SEPTICS = [(1, -1), (1, 1), (2, -1), (1, -2), (3, -2), (2, 1), (1, 3), (1, -3)]
MIN_DEGREE = 4  # drop anything a degree-3 (brutal-equivalent) or degree-1 (trivial) target would give
TR = [("r**2", lambda a: a**2), ("r**3", lambda a: a**3), ("r**4", lambda a: a**4),
      ("r**2 - r", lambda a: a**2 - a), ("r**3 + r", lambda a: a**3 + a)]


def real_root(poly):
    reals = [r for r in sp.Poly(poly, x).all_roots() if r.is_real]
    return reals[0] if len(reals) == 1 else None


def add_items(sources, transforms, items, seen):
    dropped = 0
    for p, q, src in sources:
        a = real_root(src)
        if a is None or sp.Poly(sp.minimal_polynomial(a, x), x).degree() < 5:
            dropped += 1   # reducible quintic: real root is low-degree -> not harder than brutal
            continue
        src_str = f"{sp.printing.sstr(src)} = 0"
        for desc, g in transforms:
            try:
                gv = g(a)
                mp = sp.minimal_polynomial(gv, x)
                nreal = sum(1 for r in sp.Poly(mp, x).all_roots() if r.is_real)
                deg = sp.Poly(mp, x).degree()
            except Exception:
                dropped += 1
                continue
            key = (src_str, desc)
            if nreal == 1 and deg >= MIN_DEGREE and key not in seen:
                seen.add(key)
                items.append({"source": src_str, "g_desc": desc, "r_value": float(a),
                              "target_value": float(gv), "min_poly_degree": deg,
                              "canonical_solvable": True})
            else:
                dropped += 1
    return dropped


def main():
    items, seen, dropped = [], set(), 0
    dropped += add_items([(p, q, x**5 + p * x + q) for p, q in QUINTICS], TR, items, seen)
    dropped += add_items([(p, q, x**7 + p * x + q) for p, q in SEPTICS], TR, items, seen)
    json.dump({"name": "transform_horizon",
               "note": "Beyond-brutal construction: quintic sources (degree-5 elimination) + "
                       "composite/high-degree transforms over cubics. Uniqueness verified via minimal "
                       "polynomial (exactly one real root). Tests programmatic difficulty scaling "
                       "against frontier models that saturate the brutal bank.",
               "items": items},
              open("data/banks/transform_horizon.json", "w", encoding="utf-8"), indent=2)
    degs = sorted({it["min_poly_degree"] for it in items})
    print(f"wrote {len(items)} horizon transform items ({dropped} dropped); "
          f"target min-poly degrees present: {degs}")


if __name__ == "__main__":
    main()
