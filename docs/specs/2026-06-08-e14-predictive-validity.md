# E14 — Predictive (incremental) validity of IGS

**Status: RUN (2026-06-08) — H_pred REJECTED.** Closes rebuttal #7 (no predictive/external validity)
and operationalizes the construct definition in paper §3.7.

**Outcome:** N=13. Y1 (construction): ρ(IGS,Y)=+0.55, ρ(AIME,Y)=+0.66, **partial ρ(IGS,Y|AIME)=−0.20**.
Y2 (verification): ρ(IGS,Y)=+0.72, ρ(AIME,Y)=+0.87, **partial=−0.42**. AIME predicts both outcomes at
least as well as IGS; controlling AIME, IGS adds nothing → **no incremental predictive validity**. Result
in `data/results/predictive_results.json`; written up as paper §3.8 / H9 (rejected). Caveats: permissive
IGS (strict untested), N=13 wide CIs, outcomes are solving-loaded. See §7 below.

## 1. The question

E10 (convergent) showed IGS↔AIME ρ=0.93; E13 (discriminant) showed it is math-specific, not general
capability. Both are *concurrent* correlations with another benchmark. A skeptic still asks: **"so IGS
is a redundant proxy for AIME — why not just use AIME?"** To answer that, IGS must show **predictive
value beyond forward solving**: it must predict an outcome we care about *that a forward-solving score
does not already explain*.

## 2. The construct being tested (from §3.7)

We model **IGS ≈ (forward solving) + (construction / self-verification residual)**. The residual is the
part E10's ρ=0.93 does not share with AIME: designing a formal object to an exact spec and verifying its
own correctness/uniqueness. **E14 asks whether that residual has real predictive power**: does IGS predict
construction- and verification-heavy outcomes *after controlling for forward solving (AIME)*?
This is **incremental criterion validity** — the strongest form feasible without an expensive external
ground truth (which remains future ecological-validity work).

## 3. Hypothesis & pre-registered decision rule

- **H_pred:** IGS predicts a construction/verification outcome Y **incrementally over AIME**.
- **Measure:** partial Spearman `ρ(IGS, Y | AIME)` (control = forward solving) + the pair `ρ(IGS,Y)` vs
  `ρ(AIME,Y)`. (`analysis.partial_spearman` already exists.)
- **Validity guard (from E13's lesson):** Y must SPREAD the cohort (distinct values ≥ 4, range ≥ 0.15);
  a saturated Y is an invalid outcome → INCONCLUSIVE, not a verdict.
- **SUPPORTED iff** `ρ(IGS, Y | AIME) ≥ 0.30` AND the gap `ρ(IGS,Y) − ρ(AIME,Y) ≥ 0.10` AND Y not
  saturated. **REJECTED** (IGS adds nothing beyond forward solving → "contamination-proof forward
  proxy") if partial ≈ 0. Otherwise INCONCLUSIVE.

## 4. Outcome tasks Y (machine-graded, NO LLM judge; distinct from pose/transform)

Two outcomes triangulate the two residual components. Both are new banks, sympy-verified.

### Y1 — Constrained construction (loads on "design to exact spec")
Each item gives **several simultaneous exact constraints**; the model constructs one object; sympy checks
all constraints. Distinct from pose (single target) — it is multi-constraint synthesis. Examples:
- "Construct a polynomial p(x), integer coefficients, **degree exactly 4**, **p(1)=0**, **exactly two
  distinct real roots**, **leading coefficient 1**." → sympy: degree, integer coeffs, p(1)==0, #distinct
  real roots==2, lead coeff==1.
- "Construct a rational function with a **zero at x=−1**, a **vertical asymptote at x=2**, and
  **horizontal asymptote y=3**."
- "Construct a system of two linear equations in x,y whose **unique** solution is **(√2, 1)**."
Grading: fraction of items where ALL constraints hold (a per-model Y1 score in [0,1]). Bank ~20 items,
difficulty-tiered so the cohort spreads.

### Y2 — Self-verification / error detection (loads on "verify own work")
Present K (statement, candidate answer) pairs, some true some false; the model labels each correct/incorrect.
Score = labeling accuracy. Tests component ② (checking math), which IGS implicitly requires. Examples:
- "Is x=3 the UNIQUE real solution of `x**3 - x**2 - 9*x + 9 = 0`? (no — also x=±... )" yes/no.
- "Does `(x-1)**3 + (x-1) - 1 = 0` have unique real root r+1 where r solves `x**3+x-1=0`? yes/no."
Bank ~24 balanced true/false; exact-letter machine grading. (Caveat: Y2 is closer to forward solving, so
its incremental signal over AIME is expected to be smaller than Y1's — reported, not hidden.)

## 5. Cohort & procedure

- **Models:** the E10 cohort (the 13 models that already have IGS + AIME), so all three axes (IGS, AIME, Y)
  exist on the same models. Run Y1, Y2 via the engine adapters (temp 0, reasoning cap), exact-match /
  sympy grading, like `run_discriminant.py`.
- **IGS used:** prefer the **strict** IGS (`--pose-random --pose-no-trivial`) since §3.7 shows that
  sharpens the construct (permissive IGS is muddied by restatement). If only permissive IGS is available
  for the cohort, report both and flag.

## 6. Analysis

For each Y ∈ {Y1, Y2}: `ρ(IGS,Y)`, `ρ(AIME,Y)`, `ρ(IGS,Y | AIME)` with bootstrap CIs; Williams test on
`ρ(IGS,Y) vs ρ(AIME,Y)`; apply the §3 rule. Output `data/results/predictive_results.json` with the verdict.

## 7. Threats to validity (honest)

- **Incremental, not ecological.** Y1/Y2 are still math, machine-graded — this tests whether IGS predicts
  *other construction/verification tasks* beyond forward solving, NOT real-world downstream utility or
  expert judgment (those need external outcomes; still §4.3 future work).
- **Shared method variance.** Y1 and IGS are both "construction," so a positive result partly reflects a
  shared construction factor — which is exactly the residual we claim, but it is *not* an independent
  domain. Y2 (verification) is the more independent test; report both.
- **Small N (~13)**, wide CIs; saturation guard required (Y1 must spread the cohort).

## 8. Implementation sketch (when run)

- `data/banks/construct_constraints.json` (Y1), `data/banks/verify_pairs.json` (Y2) — sympy-checkable,
  answer keys verified by construction (cf. `gen_logic_bank.py`).
- `scripts/run_predictive.py` — runs Y1/Y2 on the E10 cohort, loads IGS+AIME from `e10_results.json`,
  computes the §6 analysis, writes the verdict. Reuses `analysis.partial_spearman`, `bench_core.map_concurrent`.
- Paper: promote to **§3.8 / H9** on a SUPPORTED verdict; otherwise record honestly (REJECTED → narrow the
  claim to "contamination-proof forward proxy"; INCONCLUSIVE → keep as open).
