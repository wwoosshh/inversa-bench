# Inversa: A Contamination-Resistant, Self-Scaling Benchmark of Mathematical Ability via Problem *Construction*

*Working paper — all numbers are real measurements from this repository (branch `scoring-validity`).
Figures are generated from the result files by `scripts/make_figures.py`.*

---

## Abstract

Standard AI math benchmarks ask a model to **solve** a problem (problem → answer). Two well-known
weaknesses follow: (1) fixed public test sets **leak into training**, inflating scores above genuine
reasoning (*contamination*), and (2) each benchmark eventually **saturates** as models improve,
forcing ever harder human-authored problems. We study the **inverse** task — given an answer, ask the
model to **construct a problem** that yields it — and ask whether it is a more trustworthy basis for
measurement. We define a machine-verified score, the **Inversa Generative Score (IGS)**, computed
with a sympy oracle and **no LLM judge**. Across 22 models from 8 families we find: (i) forward
scores are measurably inflated by contamination on our own models (**+3.2%**, 95% CI [+1.0, +5.7];
GSM8K vs the fresh GSM-Symbolic), whereas the inverse task shows **no systematic gap** (+0.0%,
[−10.5, +11.0]) and is contamination-immune by construction; (ii) IGS **strongly agrees** with a
hard, non-saturated forward benchmark (AIME) — **Spearman +0.93** [+0.65, +1.0] — so it is a
*validated measure of math ability*, **not** a distinct capability (the original "generation ≠
solving" hypothesis is refuted); (iii) IGS difficulty is **self-scaling**: raising construction
difficulty (Möbius → polynomial → degree-4/composite root transforms) breaks the top ceiling so that
**no model reaches 100%** (the best, opus-4.8, tops out at 88%). We report several hypotheses we
tested and **rejected** along the way. Conclusion: problem construction is not a *separate* ability,
but it is a **contamination-resistant, auto-scaling, machine-verified way to measure the same math
ability that gold-standard forward benchmarks measure** — its durable advantage is structural, not a
new axis.

---

## Part 1 — 起 / Setup: why look at the inverse of solving

### 1.1 Two problems with "solve the problem" benchmarks

A forward benchmark gives a problem and grades the model's answer. This is the dominant paradigm
(GSM8K, MATH, AIME, …). It has two documented failure modes:

- **Contamination.** Public test sets are posted online and ingested during pre-training. The model
  can then score well by *recall* rather than reasoning. This is established empirically: GSM1k
  (Zhang et al., Scale AI, 2024, arXiv:2405.00332) built a fresh GSM8K-parallel set and found an
  overfitting gap up to **13%** for some model families; a single test-set replica in pre-training
  drastically lowers MATH loss; rephrased-sample contamination evades n-gram detection
  (arXiv:2311.04850).
- **Saturation / the treadmill.** As models improve, each benchmark's top fills with near-100%
  scores and can no longer rank the best models — like a 100-cm ruler that cannot tell apart people
  taller than 100 cm. The field responds by authoring harder benchmarks (AIME → FrontierMath), an
  expensive human treadmill, and each new benchmark eventually leaks too.

### 1.2 The idea

Reverse the direction. Instead of *"here is a problem, give the answer,"* ask *"here is the answer,
construct a problem that has it."* Two intuitions motivate this:

- A problem **constructed fresh at test time cannot be contaminated** — there is no fixed item to
  leak.
- Construction difficulty can be **raised programmatically** (compose transforms, add constraints)
  rather than by human authoring.

The empirical questions are whether construction is (a) *real* (not itself recall), (b) *measurable
and consistent*, (c) a *trustworthy* and *durable* basis for ranking models. The rest of this paper
tests these directly — including hypotheses that turned out to be **wrong**.

### 1.3 Hypotheses and verdicts (roadmap)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Forward scores are inflated by contamination | **Supported** | E12: +3.2% [+1.0,+5.7], 8/9 models |
| H2 | The inverse task is recall-resistant | **Supported** | structural + E12b: +0.0% [−10.5,+11.0] |
| H3 | Inverse "generation" is a capability **distinct** from solving (θ_gen ⟂ θ_solve) | **Rejected** | E10: IGS↔AIME +0.93 (it *tracks* ability) |
| H4 | Forward benchmarks are saturated **in general** | **Rejected / scoped** | hard forward (AIME) is not saturated |
| H5 | Forward solving here is *not* memorized (early over-claim) | **Rejected** | weak test; corrected by H1/E12 |
| H6 | IGS measures the same ability as a gold-standard hard forward test | **Supported** | E10: +0.93 [+0.65,+1.0] |
| H7 | IGS difficulty can be scaled to discriminate the frontier | **Supported** | E8: ceiling broken, opus 88% |

The honest narrative is: the *strong* form of the original vision (a brand-new ability axis) is
**false**, but a *more defensible* form (a contamination-proof, self-scaling, validated measure)
**holds**.

---

## Part 2 — 承 / Method: exactly how Inversa works

This section is deliberately concrete so a first-time reader can see every step: what we send the
model, what it returns, and how a machine (not a human, not another LLM) decides if it is correct.

![Inversa scoring pipeline](figures/fig_pipeline.png)

### 2.1 The two construction tasks

**Task A — "Pose" (construct from a target).** We give the model a target *answer* and ask it to build
an equation whose **unique** real solution is exactly that target. The exact prompt is:

```
Construct a single-variable equation in x whose UNIQUE real solution is exactly {target}.
There must be no other real solution. Radicals, fractions, exponentials and logs are allowed
and encouraged.
You may reason first, but you MUST end with the equation on its own final line, prefixed by
'#### ', in sympy syntax. Example: '#### x**2 - 2 = 0'.
```

Targets range from easy (`3`) to deliberately un-memorizable (`sqrt(2) + 1`, `the real root of
t**3 - t - 1 = 0`, `sqrt(2) + sqrt(3)`). Why irrational targets matter: the obvious template for
`sqrt(2)+1` is its minimal polynomial `x**2 - 2x - 1 = 0`, but that **also** has the conjugate root
`1 - sqrt(2)` → it fails "unique," so a model must genuinely construct (e.g. `x = sqrt(2) + 1`, or a
domain-restricted form), not pattern-match.

**Task B — "Transform" (the recall-proof core).** We generate a **random** cubic with one ugly
(irrational) real root *r*, show it to the model, and ask it to build a *new* equation whose unique
real solution is a function *g(r)* — without computing *r* numerically. Exact prompt:

```
Let r be the UNIQUE real solution of this equation:
{source}                                   e.g.  x**3 + x - 1 = 0
Do NOT solve for r numerically. By transforming the structure, construct a NEW single-variable
equation in x whose unique real solution is exactly {g}  (where r is that solution above).
You may reason first, but you MUST end with the new equation on its own final line prefixed by
'#### ', in sympy syntax. Example: '#### (x-1)**3 + (x-1) - 1 = 0'.
```

For `g = r + 1` the structural answer is the substitution `x → x − 1`: `(x-1)**3 + (x-1) - 1 = 0`.
Because the source is **random** and *r* is irrational, **no memorized problem can supply the answer**
— the output must be a function of the input shown at test time. This is what makes Task B
contamination-proof by construction.

### 2.2 Extracting the equation from the reply

Models reply in varied formats. `extract_equation` (in `tasks/posing.py`) is robust:

1. Prefer the text after the last `####` marker.
2. Otherwise take the **last line that actually parses** as an equation.
3. Clean cosmetic noise: strip LaTeX (`$`, `\(`), code fences, a leading `Final:`/`f(x) =` label;
   convert `^` → `**` (sympy reads `^` as XOR); normalize unicode (`x³` → `x**3`, `−`/`×`/`√`).
4. If the reply contains no parseable equation, **re-prompt once** for just the bare equation.

This matters: without it, a correct-but-verbose model is scored as a false zero (we observed a model
drop to 0% purely from format non-compliance before this fix).

### 2.3 Verification — a sympy oracle, no LLM judge

`verify_equation(equation, target)` (in `verifiers/math_equation.py`) decides correctness mechanically:

1. **Parse** `lhs = rhs`. Split on a single `=`; reject inequalities and relational syntax
   (`Eq(...)`). Parse each side with sympy's `parse_expr` in a **sandbox** (`__builtins__` removed)
   so untrusted model output cannot execute code.
2. **Require x to be present** (reject equations that do not constrain x).
3. **Solve** `sp.solve(lhs - rhs, x)` under a **10-second timeout** (a daemon thread; sympy can
   otherwise hang forever on adversarial input — this once stalled a run for 46 minutes).
4. Keep the **real** solutions. `valid` = the target is among them; `unique` = the real solution set
   is exactly `{target}` (within 1e-6).
5. **Numeric fallback.** If symbolic solving cannot confirm the target (e.g. a transcendental
   embedding like `exp(x**3 - x - 1) = 1`), evaluate the function on a grid + bisection to confirm
   membership and count real roots. This rescues valid constructions sympy cannot symbolically solve.

A worked example. Model output `Reasoning…\n#### x**3 - 27 = 0`, target `3`: extract →
`x**3 - 27 = 0`; solve → roots `{3, complex, complex}`; real roots `{3}`; `valid = True`,
`unique = True` → **counts**. Output `x**2 - 9 = 0`, target `3`: real roots `{3, -3}` →
`unique = False` → **does not count** (it also solves to −3). Everything is reproducible and
inspectable; the verifier has 118 passing unit tests.

### 2.4 The score: IGS

For a model we run a fixed bank of pose targets and transform items and compute two validity rates,
each in [0, 1]:

- **pose validity** = fraction of pose items meeting the *unique-solution* constraint;
- **transform validity** = fraction of transform items whose unique real root equals *g(r)*.

> **IGS = mean(pose validity, transform validity).**

IGS is **generative-only**; forward solving is never part of it (it is reported separately as a
saturation reference). The benchmark engine is `python -m inversa.cli_bench --models ...`, which
outputs a ranked IGS leaderboard (JSON + HTML) with full per-item evidence.

### 2.5 Models, data, infrastructure

- **22 models, 8 families** (Anthropic, OpenAI, Google, Meta, Mistral, DeepSeek, Qwen, Cohere,
  Amazon), weak (llama-3.1-8b, command-r) through frontier (opus-4.8, qwen3.7-plus, deepseek), via
  OpenRouter. Per-run N varies (12–22) as the roster grew and rate limits dropped individual models.
- **Banks** (all sympy-verified unique): pose targets, transform items (random ugly-root cubics ×
  {affine, reciprocal, Möbius} maps), plus harder polynomial transforms for §5.4. Forward
  comparisons use **GSM8K** and **GSM-Symbolic** (HuggingFace) and **AIME 2024+2025**.
- Runs write results **incrementally** and enforce an overall deadline, so a hang or a network drop
  never discards completed models.

---

## Part 3 — 轉 / Findings (and where the story turned)

### 3.1 Forward scores ARE inflated by contamination (H1) — and our first over-claim (H5)

We first ran a weak self-made perturbation test (number-swaps on self-authored algebra) and *wrongly
concluded* forward solving was "genuine, not memorized." That was a mistake: self-authored items
**cannot** be contaminated, so finding no gap was uninformative. We corrected it with the proper test:

For 80 templates we compared accuracy on the **GSM8K original** (in training corpora = contaminated)
vs the matched **GSM-Symbolic** instantiation (same difficulty, fresh numbers, canary-marked). The
gap is forward score inflation.

![Forward contamination gap](figures/fig_forward_gap.png)

**Result (N=9): mean gap +3.2%, 95% CI [+1.0%, +5.7%] (excludes 0); 8 of 9 models positive.** Largest
for the weakest model (llama-3.1-8b **+11%**), small for the frontier (opus, mistral-small **+1%**) —
exactly the model-dependent pattern GSM1k reports. So **H1 holds on our own models**: seeing the exact
items inflates forward scores. (We *scope* this honestly: the effect is real but modest and uneven;
we do **not** claim forward scores are broadly "fake.")

### 3.2 The inverse task has no contamination gap (H2)

By construction, Task B uses randomized inputs, so no fixed item can leak. Empirically (E12b), we
compared transform validity on **familiar/textbook** source equations (∛2, the plastic number,
Newton's `x³−2x−5`) vs **random** sources: **mean gap +0.0%, 95% CI [−10.5%, +11.0%]** (N=7) — no
systematic familiarity advantage. The CI is wide (small sample), so the **structural** argument is
primary and the measurement is corroborating. The head-to-head is the asymmetry: **forward +3.2%
(CI excludes 0) vs inverse +0.0% (CI includes 0).**

### 3.3 The big turn: IGS is *not* a separate ability — it agrees with a hard forward test (H3 rejected, H6 supported)

Our original vision claimed problem-construction is a **distinct** capability that *dissociates* from
solving. To test it we needed a hard forward benchmark that is **not** saturated, and compared the
model rankings. We used **AIME 2024+2025** (integer answers → exact grading; frontier models range
widely).

![IGS vs AIME](figures/fig_igs_vs_aime.png)

**Result (N=13): Spearman(IGS, AIME) = +0.93, 95% CI [+0.65, +1.0].** This is the central turn of the
paper:

- The **strong** hypothesis (H3, "generation is a separate axis") is **rejected** — a +0.93 rank
  correlation means IGS *tracks* general math ability, it does not measure something orthogonal.
- The **useful** hypothesis (H6) is **supported** — IGS ranks models almost identically to a
  gold-standard, expensive, human-authored hard benchmark. IGS is therefore a *validated measure of
  math ability*.

We also reject **H4** ("forward saturates in general"): AIME is *not* saturated — it discriminates
models with large headroom. So Inversa's value cannot be "forward is dead"; its value is being a
**contamination-immune, auto-generated** route to the same ranking.

### 3.4 Difficulty is self-scaling: breaking the top ceiling (H7)

On the easy transform bank the top **saturates** — three models tie at IGS 1.0, so the very top is
unmeasurable (the ruler is too short). Can we lengthen the ruler *within verifiable mathematics*? We
raised construction difficulty in two steps, each verifiable via the minimal polynomial:

- **hard:** polynomial transforms (root = r², r³) — require power-raising/elimination, not a single
  substitution;
- **brutal:** degree-4 / composite transforms (r⁴, r²+r, r³−r) — deeper elimination.

![Difficulty escalation](figures/fig_escalation.png)

**Result:** each step resolves more of the top. On the brutal bank **no model reaches 100%** — the
best (opus-4.8) tops out at **88%**, with the field spread **12%–88%**. Recomputing IGS with the hard
transform (E10-redux, N=11) **fixes the top-resolution gap** that E10 found: the number of models tied
at the maximum drops from 2 to **1** (matching AIME), with **more** distinct levels (10 vs AIME's 9)
and rank-agreement maintained (**+0.89** [+0.47, +0.99]). The reordering is informative, not uniform:
models that do simple substitution but not elimination collapse (llama-3.3-70b 77% → 6%), showing the
hard bank measures a *deeper* construction skill.

Thus the construction axis has real **headroom above saturation**, reachable programmatically and
within the verifiable-math limit — the key to future-proofing (§4.3). One caveat: separating the very
top two (opus vs qwen3.7-plus) was **not** achieved, because qwen3.7-plus (a slow reasoning model)
times out on the hard bank; this is an infrastructure limit, not a limit of the method.

### 3.5 The leaderboard

![IGS leaderboard](figures/fig_leaderboard.png)

The full IGS leaderboard (N=21, easy bank) ranks weak open models near 0 and frontier models near 1.0;
the hard/brutal banks (§3.4) re-rank and separate the top.

---

## Part 4 — 結 / Conclusion

### 4.1 What Inversa is — and is not

- **Is:** a **machine-verified** (sympy, no LLM judge), **contamination-resistant** (forward +3.2% vs
  inverse ≈0), **self-scaling** (Möbius → r² → r⁴ breaks the ceiling), **validated** (ranks like AIME,
  ρ=0.93) measure of mathematical ability.
- **Is not:** a *new, separate* capability (H3 rejected — it tracks general ability), nor inherently a
  *sharper* discriminator than a hard forward benchmark (comparable at matched difficulty).
- **Durable advantage (structural):** *verifying* a constructed problem is cheap even when
  *constructing* it is hard (a P-vs-NP-style asymmetry). So harder Inversa items cost a few lines of
  code and stay contamination-free, whereas harder forward items require expensive human authoring and
  eventually leak.

### 4.2 Threats to validity (honest)

- Moderate N (12–22) with ties → correlations carry bootstrap CIs but remain wide.
- Single domain (univariate algebra); generality untested (→ §4.3).
- The contamination gap mixes leakage with surface-fragility (we did not separate them).
- The inverse-gap CI is wide; the structural claim is primary.
- Cheap verification holds for the *algebraic* regime; transcendental uniqueness can be expensive or
  undecidable (mitigated by timeouts + numeric fallback).
- Extreme-difficulty measurement is latency-bound for slow reasoning models (qwen3.7-plus).

### 4.3 What remains for a stronger paper

- **E4 — generality:** replicate beyond algebra (number theory, systems, proofs) to show the construct
  is not algebra-specific.
- **E5 — predictive validity:** show IGS predicts an external outcome (held-out hard set, expert
  ratings).
- **E2 — reliability:** test-retest stability and inter-task consistency.
- **Larger N** to tighten all CIs; separate the very top with one more difficulty rung.

### 4.4 One-sentence conclusion

> Problem *construction* is not a separate ability, but — because it is machine-verifiable,
> contamination-immune, and cheap to scale while a hard forward benchmark is not — it is a durable,
> trustworthy instrument that measures the same mathematical ability that gold-standard forward
> benchmarks measure, and keeps discriminating models as they improve.

---

## Reproducibility

Repository `wwoosshh/inversa-bench`, branch `scoring-validity`. Engine: `python -m inversa.cli_bench`.
Core: `tasks/structural.py`, `verifiers/math_equation.py`, `scoring.py`. Experiments:
`scripts/run_forward_gap.py` (E12), `run_inverse_gap.py` (E12b), `run_e10.py` + `e10_redux.py` (E10),
`run_transform_hard.py` + `gen_transform_{hard,brutal}_bank.py` (E8). Figures: `scripts/make_figures.py`.
All scores are sympy-verified; 118 unit tests cover the verifier and scoring. Result JSONs are under
`data/results/`; banks under `data/banks/`.

**References.** GSM-Symbolic — Mirzadeh et al., Apple, ICLR 2025 (arXiv:2410.05229). GSM1k / "A Careful
Examination…" — Zhang et al., Scale AI, 2024 (arXiv:2405.00332). Rephrased-sample contamination —
arXiv:2311.04850.
