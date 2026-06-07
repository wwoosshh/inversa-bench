"""Expand the pose bank to 30 diverse structural targets (integers, rationals, quadratic/cubic
irrationals, sums/nested surds, golden ratio). Finer granularity (1/30 vs 1/6) cuts pose noise.
Each target_value is computed by sympy; the pose task verifies the model's equation against it.
Run: python scripts/gen_pose30.py
"""
import json

import sympy as sp

t = sp.Symbol("t")

def root(expr_str, guess):
    return float(sp.nsolve(sp.sympify(expr_str), t, guess))

T = [
    # integers / large
    ("3", 3.0, "integer"), ("7", 7.0, "integer"), ("12", 12.0, "integer"),
    ("100", 100.0, "large-integer"), ("4817", 4817.0, "large-integer"), ("2024", 2024.0, "large-integer"),
    # rationals
    ("7/3", 7/3, "rational"), ("11/4", 11/4, "rational"), ("22/7", 22/7, "rational"), ("5/6", 5/6, "rational"),
    # quadratic irrationals
    ("sqrt(2) + 1", float(sp.sqrt(2)+1), "quadratic-irrational"),
    ("sqrt(3) - 1", float(sp.sqrt(3)-1), "quadratic-irrational"),
    ("sqrt(5) + 2", float(sp.sqrt(5)+2), "quadratic-irrational"),
    ("2*sqrt(2)", float(2*sp.sqrt(2)), "quadratic-irrational"),
    ("sqrt(7)", float(sp.sqrt(7)), "quadratic-irrational"),
    ("3 - sqrt(2)", float(3-sp.sqrt(2)), "quadratic-irrational"),
    # the golden ratio
    ("the golden ratio (1+sqrt(5))/2", float((1+sp.sqrt(5))/2), "golden"),
    # cube roots / cubic-irrational
    ("2**(1/3) (cube root of 2)", float(sp.Rational(2)**sp.Rational(1,3)), "cube-root"),
    ("5**(1/3)", float(sp.Rational(5)**sp.Rational(1,3)), "cube-root"),
    ("the real root of t**3 - t - 1 = 0", root("t**3 - t - 1", 1.3), "cubic-irrational"),
    ("the real root of t**3 + t - 3 = 0", root("t**3 + t - 3", 1.2), "cubic-irrational"),
    ("the real root of t**3 + 2*t - 5 = 0", root("t**3 + 2*t - 5", 1.3), "cubic-irrational"),
    ("the real root of t**3 - 2*t - 5 = 0", root("t**3 - 2*t - 5", 2.1), "cubic-irrational"),
    # sums of surds
    ("sqrt(2) + sqrt(3)", float(sp.sqrt(2)+sp.sqrt(3)), "sum-of-surds"),
    ("sqrt(5) + sqrt(2)", float(sp.sqrt(5)+sp.sqrt(2)), "sum-of-surds"),
    ("sqrt(3) + sqrt(7)", float(sp.sqrt(3)+sp.sqrt(7)), "sum-of-surds"),
    # nested radicals
    ("sqrt(1 + sqrt(2))", float(sp.sqrt(1+sp.sqrt(2))), "nested-radical"),
    ("sqrt(2 + sqrt(3))", float(sp.sqrt(2+sp.sqrt(3))), "nested-radical"),
    # mixed
    ("1 + sqrt(2) + sqrt(3)", float(1+sp.sqrt(2)+sp.sqrt(3)), "sum-of-surds"),
    ("(3 + sqrt(5))/2", float((3+sp.sqrt(5))/2), "quadratic-irrational"),
]
assert len(T) == 30, len(T)
json.dump({"name": "struct_pose_targets30",
           "note": "30 structural/irrational pose targets (finer granularity than the 6-item bank).",
           "targets": [{"target_desc": d, "target_value": v, "novelty": n} for d, v, n in T]},
          open("data/banks/struct_pose_targets30.json", "w", encoding="utf-8"), indent=1)
print(f"wrote 30 pose targets")
