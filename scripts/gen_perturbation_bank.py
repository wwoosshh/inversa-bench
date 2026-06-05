"""Experiment F (cheap signal): isomorphic-perturbation fragility test for the FORWARD/solve
premise. Each template yields K structurally-identical instances differing only in constants
(and thus the answer), all sympy-verified unique-integer-root. A genuine reasoner should be
~flat (near-100%) across a template's instances; fragmented accuracy on equi-difficult fresh
variants is evidence the 'solve' success is not robust reasoning (shortcut/memorization).
Run: python scripts/gen_perturbation_bank.py
"""
import json

import sympy as sp

from inversa.verifiers.math_equation import verify_equation

X = sp.Symbol("x")


def sstr(e):
    return sp.printing.sstr(sp.expand(e))


def linear():
    out = []
    for (a, b, r) in [(2, 1, 3), (3, 2, 5), (5, -3, 7), (4, 5, 9), (7, 1, 4), (6, -2, 8)]:
        out.append((f"{a}*x + {b} = {a * r + b}", r))
    return out


def disguised_cube():
    return [(f"{sstr((X - r)**3)} = 0", r) for r in (2, 3, 4, 5, 7, 6)]


def factored_complex():
    # (x - r)(x^2 + x + 1) = 0 -> unique real root r (quadratic is irreducible over R), expanded
    return [(f"{sstr((X - r) * (X**2 + X + 1))} = 0", r) for r in (3, 4, 5, 7, 8, 11)]


def radical_extraneous():
    # x + sqrt(x + a) = b with a = k^2 - r, b = r + k -> root r, plus an extraneous root from squaring
    out = []
    for (r, k) in [(2, 3), (5, 2), (3, 4), (7, 3), (4, 5), (9, 2)]:
        a, b = k * k - r, r + k
        out.append((f"x + sqrt(x + {a}) = {b}", r))
    return out


TEMPLATES = {
    "T1_linear": linear(),
    "T2_disguised_cube": disguised_cube(),
    "T3_factored_complex": factored_complex(),
    "T4_radical_extraneous": radical_extraneous(),
}

problems, dropped = [], 0
for tname, items in TEMPLATES.items():
    kept = 0
    for eq, ans in items:
        v = verify_equation(eq, ans)
        if v.valid and v.unique:
            problems.append({"equation": eq, "answer": ans, "template": tname})
            kept += 1
        else:
            dropped += 1
            print(f"DROP [{tname}] {eq} valid={v.valid} unique={v.unique}")
    print(f"{tname}: {kept} kept")

json.dump({"name": "perturbation_set",
           "note": "Isomorphic instances per template (same structure, perturbed constants, "
                   "sympy-verified unique integer root). Within-template accuracy spread tests "
                   "robustness of forward solving vs surface/shortcut fragility.",
           "problems": problems},
          open("data/banks/perturbation_set.json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote {len(problems)} problems across {len(TEMPLATES)} templates ({dropped} dropped)")
