"""Ceiling-break solve bank: HARD integer-answer problems (clean, no precision noise) chosen
so even strong models slip on at least one -> goal: <=1 model scores 100%. Mix of extraneous-
root radical traps, multi-radical, disguised perfect powers, and transcendental monotonic forms.
Auto-keeps only sympy-verified UNIQUE-real-root problems. Run: python scripts/gen_solve_hard2.py
"""
import json

from inversa.verifiers.math_equation import verify_equation

CANDIDATES = [
    ("sqrt(x) + sqrt(x + 9) = 9", 16, "multi-radical"),
    ("sqrt(x + 40) - sqrt(x) = 4", 9, "multi-radical"),
    ("sqrt(x + 15) + sqrt(x) = 15", 49, "multi-radical"),
    ("x + sqrt(x + 7) = 5", 2, "extraneous"),
    ("sqrt(x + 4) + sqrt(x - 1) = 5", 5, "multi-radical"),
    ("sqrt(2*x + 8) - sqrt(x + 5) = 1", 4, "extraneous"),
    ("sqrt(2*x + 6) + sqrt(x + 11) = 2*x - 2", 5, "radical-quadratic"),
    ("sqrt(x - 1) + sqrt(2*x + 6) = 6", 5, "extraneous"),
    ("x**3 - 6*x**2 + 12*x - 8 = 0", 2, "disguised-cube"),
    ("x**3 - 9*x**2 + 27*x - 27 = 0", 3, "disguised-cube"),
    ("(x**2 - 4*x + 4)*(x**2 + 1) = 0", 2, "disguised-factored"),
    ("x**3 + 3*x**2 + 3*x + 1 = 8", 1, "disguised-cube"),
    ("x**6 - 2*x**3 + 1 = 0", 1, "disguised-power"),
    ("x**5 + x**3 + x - 3 = 0", 1, "quintic-monotone"),
    ("(x - 7)*(x**2 + x + 1) = 0", 7, "factored"),
    ("2**(x + 1) + 2**(x - 1) = 20", 3, "exponential"),
    ("3**x + 3**(x + 1) = 108", 3, "exponential"),
    ("x + 2**x = 6", 2, "transcendental-monotone"),
    # NOTE: x+3**x=30 and x*2**x=8 are mathematically clean (unique integer roots) but our
    # verifier can hang on them — sp.solve enters a GIL-holding C routine that thread-based
    # timeouts cannot interrupt. Excluded so bank generation (and any re-verify) stays safe.
]

problems, dropped = [], []
for eq, ans, tag in CANDIDATES:
    r = verify_equation(eq, ans)
    if r.valid and r.unique:
        problems.append({"equation": eq, "answer": ans, "tag": tag})
        print(f"OK   [{tag:24}] {eq:44} -> {ans}")
    else:
        dropped.append(eq)
        print(f"DROP [{tag:24}] {eq:44} valid={r.valid} unique={r.unique} sols={r.solutions[:3]}")

with open("data/banks/solve_set_hard2.json", "w", encoding="utf-8") as f:
    json.dump({"name": "solve_set_hard2",
               "note": "Hard integer unique answers (clean), tuned so <=1 model maxes -> the top "
                       "of the solve axis becomes measurable. sympy-verified unique-real.",
               "problems": problems}, f, indent=2)
print(f"\nwrote {len(problems)} problems ({len(dropped)} dropped)")
