"""Build a CLEAN, capability-graded solve bank: integer unique answers only (no 1e-6
precision games / irrational answers that caused noise like opus<scout). Difficulty rises
easy -> hard so weak models fail the top tiers and strong models pass -> a capability-ordered
solve axis. Every problem is sympy-verified to have the stated value as its UNIQUE real root.
Run: python scripts/gen_solve_graded.py
"""
import json

from inversa.verifiers.math_equation import verify_equation

CANDIDATES = [
    ("2*x = 8", 4, "easy"),
    ("x + 5 = 12", 7, "easy"),
    ("3*x - 6 = 9", 5, "easy"),
    ("2**x = 32", 5, "moderate"),
    ("x**3 = 27", 3, "moderate"),
    ("3**(x - 1) = 81", 5, "moderate"),
    ("x + sqrt(x) = 6", 4, "moderate"),
    ("x + sqrt(x + 7) = 5", 2, "hard-extraneous"),
    ("sqrt(x + 4) + sqrt(x - 1) = 5", 5, "hard-radical"),
    ("sqrt(2*x + 8) - sqrt(x + 5) = 1", 4, "hard-extraneous"),
    ("sqrt(2*x + 6) + sqrt(x + 11) = 2*x - 2", 5, "hard-radical"),
    ("x**3 + 3*x**2 + 3*x + 1 = 8", 1, "hard-disguised"),
    ("sqrt(x + 15) + sqrt(x) = 15", 49, "hard-radical"),
    ("x**5 + x**3 + x - 3 = 0", 1, "hard-quintic"),
    ("(x - 7)*(x**2 + x + 1) = 0", 7, "hard-factored"),
]

problems, dropped = [], []
for eq, ans, tier in CANDIDATES:
    r = verify_equation(eq, ans)
    if r.valid and r.unique:
        problems.append({"equation": eq, "answer": ans, "tier": tier})
        print(f"OK   [{tier:16}] {eq:42} -> {ans}")
    else:
        dropped.append((eq, ans, r.valid, r.unique, r.solutions[:3]))
        print(f"DROP [{tier:16}] {eq:42} -> valid={r.valid} unique={r.unique} sols={r.solutions[:3]}")

with open("data/banks/solve_set_graded.json", "w", encoding="utf-8") as f:
    json.dump({"name": "solve_set_graded",
               "note": "Integer unique answers, graded easy->hard; sympy-verified unique-real. "
                       "Clean capability axis (no irrational-precision noise).",
               "problems": problems}, f, indent=2)
print(f"\nwrote {len(problems)} problems ({len(dropped)} dropped)")
