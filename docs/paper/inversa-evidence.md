# Inversa: Problem *Construction* as a Distinct, Recall-Resistant Measure of AI Mathematical Ability

> **Status: paper-preparation scaffold (v0).** This document organizes the thesis, the
> evidence gathered so far, and — critically — the evidence still required for a publishable,
> reviewer-proof claim. Sections marked **[ESTABLISHED]** rest on data we have collected;
> sections marked **[TODO/NEEDED]** are gaps that must be closed before the headline claim is
> defensible. Honesty here is strategic: an overclaiming draft dies in review.

---

## 0. One-sentence thesis

> Measuring whether a model can, given an answer, **construct a problem that yields it** (the
> *inverse* task) is a valid indicator of mathematical ability that is **distinct from, and
> remains discriminative where, forward problem-*solving* saturates.**

We operationalize this with the **Inversa Generative Score (IGS)**, a machine-verified,
memorization-resistant score, and show that among frontier models that all score 100% on a
clean solve benchmark, IGS still spreads them across a wide range.

---

## 1. Motivation

Forward benchmarks (problem → answer) increasingly **saturate**: frontier models score at or
near 100% on school/competition algebra, so the benchmark can no longer rank the top — we
cannot tell a "100-point" model from a "150-point" model. A measure that still discriminates in
that regime would be valuable both practically (frontier evaluation) and scientifically (does
"understanding" decompose into more than recall of solutions?).

The candidate: **inverse construction.** Solving a known problem can be satisfied by retrieval
of a memorized solution path; *constructing* a problem to specification — especially under
constraints with no memorized template — appears to demand compositional manipulation of
structure. The empirical question is whether this is (a) real (not itself recall) and (b)
distinct from solving.

---

## 2. The construct and its threat model

Generation could occupy one of three levels; the thesis needs at least the top one to be
measurable and to vary across models:

1. **Specific-problem recall** — emit a memorized problem. (Would refute the thesis.)
2. **Parametric-template recall** — instantiate a memorized *family* (e.g. `(x−T)^n = 0`) with
   a free constant. Generalizes over the constant but not structure.
3. **Flexible compositional construction** — build/transform novel structure to meet a spec.

**Falsifiability line adopted:** a model demonstrates level-3 if, on inputs whose novelty we
*manufacture at test time*, it produces outputs that **verifiably preserve an invariant** —
because then no memorized artifact can satisfy the task. This is the only recall-proof bar
achievable against closed models whose training data we cannot inspect.

---

## 3. Method

### 3.1 Two recall-resistant generative tasks (sympy forward-verified)

- **Pose (Form A).** "Construct an equation whose UNIQUE real solution is exactly *T*", for
  *T* on a novelty ladder from integers to irrationals/structural targets
  (`sqrt(2)+1`, the real root of `t³−t−1`, `sqrt(2)+sqrt(3)`). The naive template — *T*'s
  minimal polynomial — carries conjugate roots and **fails uniqueness**, so success on the
  irrational rungs requires genuine construction.
- **Transform (Form B).** Given a **freshly randomized** ugly-root cubic *P* with unique real
  root *r*, "construct an equation whose unique real solution is *g(r)*" for
  *g* ∈ {`r+1`, `2r`, `1/r`, `1/(r+1)`, `r/(r+1)`} (affine, reciprocal, and composed Möbius
  maps; all preserve root-uniqueness). Because the source is random and *r* is irrational, the
  answer is a **structural function of the test-time input** — memorization cannot help. We
  verified that high scorers produce genuine substitutions (e.g. `(x−1)³+(x−1)−1=0`), **not**
  numeric plug-ins.

### 3.2 Verification

All judgments are made by a sandboxed sympy oracle (`verify_equation`): parse → solve →
check the target is the **unique** real root. A numeric fallback (bounded root scan) validates
transcendental constructions sympy cannot symbolically solve. No LLM-judge; fully glass-box —
every posed/transformed equation is recorded for human inspection.

### 3.3 The indicator

**IGS = mean(pose_validity, transform_validity)**, each ∈ [0,1] (`src/inversa/scoring.py`).
Generative-only; solving is **never** part of IGS and is reported separately as a saturation
reference.

### 3.4 Reproducibility

Code, banks, and per-item evidence reports are version-controlled (repo
`wwoosshh/inversa-bench`, branch `scoring-validity`). Key components: `tasks/structural.py`
(tasks), `verifiers/math_equation.py` (oracle), `scoring.py` (IGS),
`scripts/gen_l3_banks.py` / `gen_solve_hard2.py` (banks), `cli_structural.py` /
`cli_experiment.py` (runners), `scripts/build_leaderboard.py` (IGS leaderboard).

---

## 4. Experimental setup

- **Models:** up to 15 across 7 families (Anthropic, OpenAI, Google, Meta, Mistral, DeepSeek,
  Qwen), spanning weak (llama-3.1-8b, qwen-2.5-7b, claude-3.5-haiku) to frontier
  (opus-4.8, qwen3.7-plus, deepseek-v3.2, gpt-4o). Accessed via OpenRouter.
- **Solve axis:** graded integer-answer banks, sympy-verified unique
  (`solve_set_graded`, `solve_set_hard2`).
- **Generative axes:** pose ladder (6 targets), transform bank (30 random-cubic × Möbius items).

---

## 5. Results

### 5.1 Forward solving saturates **[ESTABLISHED]**

On the hardest clean integer-answer bank (`solve_set_hard2`, 18 items incl. extraneous-root
radical traps, disguised cubes/powers, transcendental-monotone), **6 of 12 models scored
100%**; the range was 72%–100%. Even competition-style traps do not break the ceiling for the
frontier tier. → *The solve axis cannot rank the top.*

### 5.2 IGS discriminates where solve cannot — the headline **[ESTABLISHED, suggestive]**

Among the **6 models tied at solve = 100%**, IGS ranged **0.27 – 0.92** (e.g.
gemini-3.1-flash-lite 0.92 vs gemini-3.5-flash 0.27, both solve 100%). A measure that is flat
(all 100%) on solving is spread > 3× on IGS over the identical model set. *This is the core
demonstration of the thesis's value proposition.* (Source: `igs_leaderboard.json`, N=12.)

| rank | model | IGS | pose | transform | solve (ref) |
|---|---|---|---|---|---|
| 1 | gemini-3.1-flash-lite | 0.92 | 0.83 | 1.00 | 100% |
| 2 | llama-4-scout | 0.83 | 0.83 | 0.83 | 100% |
| 3 | claude-opus-4.8 | 0.83 | 0.67 | 1.00 | 100% |
| 4 | deepseek-v3.2 | 0.73 | 0.50 | 0.97 | 94% |
| … | … | … | … | … | … |
| 11 | gemini-3.5-flash | 0.27 | 0.00* | 0.53 | 100% |
| 12 | llama-3.1-8b | 0.07 | 0.00 | 0.13 | 72% |

\*known format artifact (see §6.4).

### 5.3 Recall is refuted on the recall-proof axis **[ESTABLISHED]**

Transform is impossible to satisfy by retrieval (output pinned to a random input), yet scores
ranged 0.03–1.00 with high scorers producing verified structural substitutions. → Generative
success is genuine construction, not memorized output (refuting threat-level 1).

### 5.4 Convergent vs discriminant validity — an honest, instructive reversal **[PARTIAL]**

- **Convergent:** the generative measures intercorrelate (e.g. adversarial-validity ~ pose
  Spearman ≈ +0.76; pose ~ transform ≈ +0.47–0.60) — consistent with a single latent
  "generative" factor.
- **Discriminant (caveat):** with a *noisy* solve axis, solve↔generative looked weak
  (+0.16–0.47, apparent dissociation); after **cleaning** the solve axis the correlation rose
  (+0.61–0.87, apparent tracking). **The early "dissociation" was substantially solve-axis
  measurement noise.** The defensible signal is therefore **not** a low global correlation but
  the **within-solve-ceiling spread** of §5.2: when solving is held constant at its maximum,
  generative ability still varies. (This reversal is itself a methodological contribution: it
  shows why clean axis measurement is mandatory before claiming dissociation.)

---

## 6. Threats to validity (what a reviewer will attack — and our position)

1. **Small N, ties.** N=12 with many tied/ceilinged values → all global correlations are
   *suggestive, not conclusive*. Bootstrap CIs not yet computed. → §7.
2. **Single domain.** Univariate algebraic equations only. Construct generality untested. → §7.
3. **Ceiling confound.** Because solve saturates, we cannot cleanly separate "generative
   ability" from a general capability factor; the within-ceiling spread (§5.2) is the workaround
   but is driven by a handful of models. Generative tasks must also be hardened so the top tier
   does not ceiling on IGS either (currently 2 models at transform = 1.0). → §7.
4. **Format/extraction artifacts.** Non-format-compliant output still injects false zeros
   (gemini-3.5-flash pose = 0.00 despite real ability). IGS is noisy for verbose/non-compliant
   models. Mitigated by `####` markers + parse-aware extraction; not eliminated. → constrained
   decoding needed.
5. **IGS weighting** (equal pose/transform) is an unvalidated design choice.
6. **Single weak-solver / unstable adversarial axis** (floored when the solver-to-fool is too
   strong) — hence adversarial is currently **excluded** from IGS; documented, not hidden.
7. **Verification limits.** Numeric uniqueness check uses a bounded window (heuristic); sympy
   `solve` can hang on adversarial input (thread timeouts cannot interrupt GIL-holding C code —
   two transcendental forms were excluded). Possible (rare) misjudgments.
8. **Contamination risk.** Pose targets are fixed; transform mitigates via randomization. Banks
   committed to a public repo could leak into future training.
9. **Infra noise.** OpenRouter rate-limits / quota / connection drops caused missing models
   (e.g. gpt-4o); incremental writing now preserves partial runs.
10. **No predictive/criterion validity yet.** IGS has not been shown to predict any external
    outcome (its convergent/discriminant story is internal).

---

## 7. Evidence still required for a publication-level claim **[NEEDED — experiment plan]**

The thesis is currently *demonstrated as plausible*, not *proven*. To make it reviewer-proof:

- **E1 — Scale & statistics.** 30–50+ models across tiers/families; report Spearman with
  bootstrap 95% CIs, not point estimates; pre-register hypotheses.
- **E2 — Reliability.** Test–retest (same model twice → IGS stability) and inter-task reliability
  (do pose & transform consistently co-rank? Cronbach-style α over generative subtasks).
- **E3 — Discriminant validity, done right.** A difficulty regime where neither solve nor IGS
  ceilings, then partial correlation / residual regression: does IGS explain model variance
  **beyond** a clean solve score? (The within-ceiling spread is the qualitative version; this is
  the quantitative one.)
- **E4 — Construct generality.** Replicate across ≥3 domains (e.g. number theory, systems,
  inequality construction, possibly program/proof construction) to show it is not an
  algebra-specific quirk.
- **E5 — Predictive/criterion validity.** Show IGS predicts an *external* outcome — e.g.
  performance on a held-out hard construction set, or human-expert ratings of generated-problem
  quality — establishing it measures something useful, not just self-consistent.
- **E6 — Anti-contamination.** Freshly generate all targets/sources per run (transform already
  does); keep a private held-out bank; report results on never-published items.
- **E7 — Kill measurement artifacts.** Constrained/structured decoding (or tool-call output) so
  format compliance cannot depress scores; finish unicode/marker normalization.
- **E8 — Harden generative ceiling.** Multi-step composed transforms / conjunctive pose
  constraints so the top tier (currently IGS ≈ 0.83–0.92) is resolved.
- **E9 — Ablations & baselines.** Human baseline on a subset; trivial-templater baseline to show
  IGS > template-instantiation; sensitivity of IGS to its weighting.

---

## 8. Claims we can/can't make today

| Claim | Status |
|---|---|
| Generative construction is *real* (not pure recall) | **Supported** (recall-proof transform spreads 0–1 via genuine substitution) |
| It is *measurable* and *internally coherent* | **Supported** (gen axes intercorrelate; IGS defined & reproducible) |
| It *discriminates where solving saturates* | **Supported, suggestive** (6 solve-100% models span IGS 0.27–0.92, N=12) |
| It is a *distinct dimension* from general capability | **Not yet** (ceiling confound; needs E3) |
| It *generalizes beyond algebra* | **Untested** (needs E4) |
| It *predicts anything external* | **Untested** (needs E5) |

---

## 9. Data & artifact index (for the eventual paper)

- IGS leaderboard: `data/results/igs_leaderboard.{html,json}` (N=12).
- Macro battery: `data/results/macro3_experiment.json` (solve+adversarial),
  `macro3_level3.json` (pose+transform); earlier sweeps `macro_*`, `macro_*_v2` (solve-axis
  cleaning study, §5.4).
- Per-item glass-box reports: `*_report.html` (every equation, target, verdict).
- Banks: `data/banks/{struct_pose_targets,transform_bank,solve_set_graded,solve_set_hard2}.json`.
- Methods provenance: commit history on `scoring-validity` (verifier hardening, scoring
  decomposition, ceiling-break, IGS formalization).

---

## 10. Next action

Pick the gap to close first. Recommended order for maximal reviewer impact:
**E7 (artifacts) → E1/E2 (scale + reliability) → E3 (discriminant done right) → E4 (generality).**
E5 (predictive validity) is the highest-value but hardest; plan it once the above hold.
