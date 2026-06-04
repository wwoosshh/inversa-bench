"""Generate the level-3 banks (reproducible, sympy-verified):
  - data/banks/struct_pose_targets.json : irrational/structural targets (Form A ladder)
  - data/banks/transform_bank.json      : random-ish ugly-root cubics + root transforms (Form B)
Each Form-B entry is sanity-checked: the canonical structural answer must verify as the
UNIQUE real solution at g(r), proving the task is solvable (so a 0 score is the model's fault).
Run: python scripts/gen_l3_banks.py
"""
import json
import os

import sympy as sp

from inversa.verifiers.math_equation import verify_equation

X, R, T = sp.symbols("x r t")


def fval(expr) -> float:
    return float(sp.N(expr, 20))


# ---- Form A: structural-target ladder (nice -> un-memorable) -------------------------
plastic = sp.nsolve(T**3 - T - 1, T, 1.3)
POSE_TARGETS = [
    ("3", 3, "integer"),
    ("4817", 4817, "large-integer"),
    ("7/3", sp.Rational(7, 3), "rational"),
    ("sqrt(2) + 1", sp.sqrt(2) + 1, "quadratic-irrational"),
    ("the real root of t**3 - t - 1 = 0 (~1.3247)", plastic, "cubic-irrational"),
    ("sqrt(2) + sqrt(3)", sp.sqrt(2) + sp.sqrt(3), "sum-of-surds"),
]

# ---- Form B: ugly-root cubics (one real root) + structural root transforms ------------
# x**3 + p*x + q with 4p**3 + 27q**2 > 0 -> exactly one real root (irrational, un-memorable).
CUBICS = [(1, -1), (1, -3), (2, -5), (-1, -1), (3, -2), (1, 1)]
# g(r) as (label, lambda building the *canonical* answer equation in x, and g as a sympy fn)
TRANSFORMS = [
    ("r + 1", lambda src: src.subs(X, X - 1), lambda r: r + 1),
    ("2*r",   lambda src: src.subs(X, X / 2), lambda r: 2 * r),
    # r**2 omitted: no clean root-substitution guarantees a UNIQUE real root, so it would be
    # an unfairly/possibly-unsolvable task. Every kept transform is canonical-solvable=True.
]


def real_root(poly_expr):
    roots = [complex(s) for s in sp.solve(sp.Eq(poly_expr, 0), X)]
    reals = [c.real for c in roots if abs(c.imag) < 1e-9]
    assert len(reals) == 1, f"{poly_expr} has {len(reals)} real roots"
    return reals[0]


def build():
    pose = [{"target_desc": d, "target_value": fval(v), "novelty": nov} for d, v, nov in POSE_TARGETS]

    transform = []
    for (p, q) in CUBICS:
        src_expr = X**3 + p * X + q
        src_str = f"{sp.printing.sstr(src_expr)} = 0"
        r = real_root(src_expr)
        for (g_desc, canon, g_fn) in TRANSFORMS:
            target = float(g_fn(sp.Float(r, 20)))
            solvable = None
            if canon is not None:
                cand = canon(src_expr)
                cand_str = f"{sp.printing.sstr(sp.expand(cand))} = 0"
                solvable = verify_equation(cand_str, target).unique
            transform.append({
                "source": src_str, "g_desc": g_desc,
                "r_value": float(r), "target_value": target,
                "canonical_solvable": solvable,
            })

    os.makedirs("data/banks", exist_ok=True)
    with open("data/banks/struct_pose_targets.json", "w", encoding="utf-8") as f:
        json.dump({"name": "struct_pose_targets", "targets": pose}, f, indent=2)
    with open("data/banks/transform_bank.json", "w", encoding="utf-8") as f:
        json.dump({"name": "transform_bank", "items": transform}, f, indent=2)

    print(f"pose targets: {len(pose)}")
    for t in pose:
        print(f"  [{t['novelty']:>20}] {t['target_desc']:<45} = {t['target_value']:.6f}")
    print(f"transform items: {len(transform)} (canonical solvable checks below)")
    for it in transform:
        print(f"  {it['source']:<22} g={it['g_desc']:<6} r={it['r_value']:.4f} "
              f"target={it['target_value']:.4f} solvable={it['canonical_solvable']}")


if __name__ == "__main__":
    build()
