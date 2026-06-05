# Inversa: Problem *Construction* as a Distinct, Recall-Resistant Measure of AI Mathematical Ability

> **Status: paper-preparation scaffold (v0).** This document organizes the thesis, the
> evidence gathered so far, and — critically — the evidence still required for a publishable,
> reviewer-proof claim. Sections marked **[ESTABLISHED]** rest on data we have collected;
> sections marked **[TODO/NEEDED]** are gaps that must be closed before the headline claim is
> defensible. Honesty here is strategic: an overclaiming draft dies in review.

---

## 0. One-sentence thesis (two pillars)

> **Pillar 1 (primary — contamination-robustness).** Forward (problem→answer) benchmarks suffer
> *measurable memorization contamination* (objectively documented in the literature); the
> *inverse* task (answer→problem), and especially structural transforms of randomized inputs, is
> **recall-resistant by construction**, so it is a **more contamination-robust** way to measure
> ability.
>
> **Pillar 2 (complementary — non-saturation).** Inverse construction **still discriminates models
> where (our) forward banks saturate**, giving a frontier discriminator.

We operationalize both with the **Inversa Generative Score (IGS)**, a machine-verified,
recall-resistant score. **Status:** Pillar-1's *premise* (forward is contaminated) is
literature-established (cite); its *relative claim* (inverse contaminates less) is **ours to
prove** via a head-to-head contamination-gap test (§7, E12). Pillar-2 hinges on E10 (§5.6).

---

## 1. Motivation

**Primary motivation — contamination (literature-established).** Forward (problem→answer)
benchmarks suffer *measurable memorization contamination*: fixed public test sets leak into
training corpora, inflating scores above genuine reasoning. This is **objectively documented**,
not hypothetical:
- **GSM1k** (Zhang et al., Scale AI, 2024, arXiv:2405.00332) built a fresh GSM8K-parallel set and
  measured an overfitting gap **up to 13%** for some families (Phi, Mistral); the gap correlates
  with a model's propensity to emit GSM8K examples — direct evidence of partial memorization.
- **Replica-loss** studies show a *single* MATH test-set replica in pretraining drastically lowers
  loss; **rephrased-sample contamination** (arXiv:2311.04850) evades n-gram detection while
  inflating MMLU/GSM8K/HumanEval.

**Honest nuance:** the contamination gap is *model-dependent* — large for some families, **minimal
for frontier (GPT/Claude/Gemini)** per GSM1k. So the claim is "forward has measurable, uneven
contamination," not "all forward scores are fake."

**The candidate fix — inverse construction.** A benchmark whose items are *generated fresh at test
time* cannot be contaminated. Constructing a problem to specification — especially transforming a
*randomized* input — is **recall-resistant by construction**: no fixed item exists to leak. Hence
the thesis: **inverse is a more contamination-robust benchmark.** The literature establishes the
*premise* (forward is contaminated); **our job is the *relative* claim** — that inverse exhibits a
*smaller contamination gap* than forward on the same models (head-to-head, §7 E12). A secondary,
already-observed benefit is non-saturation (§5.6).

> **Correction note (intellectual honesty).** An earlier draft over-read our own Experiment F
> (§5.5) as "forward solving is genuine, not memorized" and pivoted the thesis to non-saturation
> only. That was wrong: Exp-F tested the *weakest* slice (number-swaps on simple computed algebra)
> and merely *agrees* with GSM1k's frontier-robustness — it does **not** refute contamination,
> which the literature establishes. The contamination pillar is therefore **restored as primary.**

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
`scripts/gen_l3_banks.py` / `gen_solve_hard2.py` / `gen_perturbation_bank.py` (banks),
`cli_structural.py` / `cli_experiment.py` / `scripts/run_perturbation.py` (runners),
`scripts/build_leaderboard.py` (IGS leaderboard), `scripts/macro_analyze.py` (correlations +
bootstrap CIs), `analysis.py` (`spearman_ci`).

---

## 4. Experimental setup

- **Models:** up to **22 across 8 families** (Anthropic, OpenAI, Google, Meta, Mistral, DeepSeek,
  Qwen, Cohere, Amazon), spanning weak (llama-3.1-8b, qwen-2.5-7b, command-r) through mid
  (gpt-4o-mini, mistral-large, nova-pro) to frontier (opus-4.8, qwen3.7-plus, deepseek-v3.2/r1).
  Accessed via OpenRouter. (Per-run N varies 12→21 as the roster grew and infra/rate-limits
  dropped individual models; partial-result writing preserves completed models.)
- **Solve axis:** graded integer-answer banks, sympy-verified unique
  (`solve_set_graded`, `solve_set_hard2`).
- **Generative axes:** pose ladder (6 targets), transform bank (30 random-cubic × {affine,
  reciprocal, Möbius} items).
- **Perturbation axis (Experiment F):** 4 templates × 6 isomorphic instances (constants
  perturbed, sympy-verified unique) — `solve_set` robustness probe.

---

## 5. Results

### 5.1 *Our* verifiable solve banks saturate at the top **[ESTABLISHED — scoped]**

On our hardest clean integer-answer bank (`solve_set_hard2`, 18 items incl. extraneous-root
radical traps, disguised cubes/powers, transcendental-monotone), **8 of the 19 evaluated models
scored 100%**; the range was 28%–100%. → *Our solve axis cannot rank the top tier.*

**Important scope (do not overclaim).** This shows *our banks* saturate, **not** that forward
benchmarks saturate in general. Old/easy public benchmarks (GSM8K, MATH) are indeed saturated
for frontier models, but deliberately-hard ones (AIME, GPQA, FrontierMath, HLE) are **not** —
they discriminate with large headroom. Forward benchmarking is a *treadmill*: each benchmark
saturates as models improve and the field authors harder ones. So IGS's value cannot rest on
"forward is saturated/broken"; it must rest on being an **orthogonal, recall-resistant,
auto-scaling** axis — and on the decision framework in §5.6, which is **not yet tested**.

### 5.2 IGS discriminates where solve cannot — the headline **[ESTABLISHED, suggestive]**

Among the **8 models tied at solve = 100%**, IGS ranged **0.65 – 1.00**. A measure that is flat
(all 100%) on solving spreads widely on IGS over the identical model set. *This is the core
demonstration of the thesis's value proposition.* (Source: `igs_leaderboard.json`, N=21.)

| rank | model | IGS | pose | transform | solve (ref) |
|---|---|---|---|---|---|
| 1 | gemini-3.1-flash-lite | 1.00 | 1.00 | 1.00 | 100% |
| 2 | claude-opus-4.8 | 1.00 | 1.00 | 1.00 | 100% |
| 3 | qwen3.7-plus | 1.00 | 1.00 | 1.00 | n/a |
| 4 | deepseek-v3.2 | 0.92 | 0.83 | 1.00 | 100% |
| 5 | claude-haiku-4.5 | 0.88 | 0.83 | 0.93 | 100% |
| … | … | … | … | … | … |
| 19 | qwen-2.5-7b | 0.18 | 0.00 | 0.37 | 89% |
| 20 | command-r-08-2024 | 0.10 | 0.17 | 0.03 | 28% |
| 21 | llama-3.1-8b | 0.07 | 0.00 | 0.13 | 50% |

### 5.3 Recall is refuted on the recall-proof axis **[ESTABLISHED]**

Transform is impossible to satisfy by retrieval (output pinned to a random input), yet scores
ranged 0.03–1.00 with high scorers producing verified structural substitutions. → Generative
success is genuine construction, not memorized output (refuting threat-level 1).

### 5.4 Convergent vs discriminant validity — with bootstrap CIs **[PARTIAL]**

(N=20 intersection of solve+IGS; Spearman, seeded percentile bootstrap 95% CI.)

- **Convergent:** the generative measures intercorrelate — pose ~ transform **+0.58 [+0.14,
  +0.85]**, adversarial-validity ~ transform **+0.54 [+0.16, +0.80]** (CIs exclude 0) —
  consistent with a single latent "generative" factor.
- **Discriminant — an instructive reversal:** with a *noisy* solve axis, solve↔generative looked
  weak (+0.16–0.47, apparent dissociation); after **cleaning** the solve axis it rose to
  solve ~ transform **+0.74 [+0.40, +0.92]**, solve ~ pose **+0.73 [+0.43, +0.89]** (tracking).
  **The early "dissociation" was substantially solve-axis measurement noise.** The defensible
  signal is therefore **not** a low global correlation but the **within-solve-ceiling spread**
  (§5.2): when solving is held at its maximum, IGS still varies 0.65–1.00. (This reversal is
  itself a methodological contribution: clean axis measurement is mandatory before claiming
  dissociation.)

### 5.5 Number-perturbation robustness on simple algebra (Experiment F) **[WEAK — scoped, not a memorization refutation]**

We ran a *number-swap* perturbation test: 4 templates × 6 **isomorphic** instances (identical
structure, perturbed constants → different answers, sympy-verified). Result (N=7): **5 of 7 models
scored 100% across all templates**; only weak models fragmented (llama-3.1-8b 62%). 

**This does NOT prove forward solving is un-memorized — we initially overclaimed it.** Against the
literature it is the *weakest* possible slice: (i) number-swaps are the perturbation with the
*smallest* effect, whereas GSM-Symbolic (Apple, ICLR 2025, arXiv:2410.05229) shows an irrelevant
*added clause* drops even frontier models **up to 65%**; (ii) our domain is simple integer algebra,
which is *computed* not memorized — real contamination evidence is on word-problems/specific items;
(iii) self-authored items mean there was no "canonical (likely-memorized)" version to compare a
fresh one against, so we measured fragility, not a contamination *gap*. Our "frontier-robust"
result merely **agrees with** GSM1k (Scale AI, arXiv:2405.00332: GPT/Claude/Gemini show minimal
GSM8K→GSM1k drop; Phi/Mistral up to **13%**) — it does not refute contamination.

**Honest status:** we did *not* establish "forward is genuine, not memorized." We showed only that
strong models compute simple algebra robustly under number-swaps (consistent with the literature).
A real refutation needs the stronger probes (added-clause distractors, word-problem domain,
canonical-vs-fresh gap, larger N) — see E11, §7. Crucially, Inversa's thesis does **not depend** on
this: its value rests on being a non-saturating, recall-resistant, auto-scaling axis (§5.6), not on
forward being "fake."

### 5.6 The value-decision framework — when is IGS worth keeping? **[DESIGN, not yet tested]**

Assume a *hard* forward benchmark that currently discriminates models well (the honest baseline,
since hard forward benchmarks are **not** saturated, §5.1). Run the same models on it; compare
the forward ranking to the IGS ranking. The outcome decides IGS's fate:

- **(A) Rankings DIFFER.** Being "different" is *not* automatically a virtue — it could be noise.
  IGS is trustworthy only if we can **objectively establish what ability it measures** and *why*
  a high-IGS / low-forward model is good at something real (construct + predictive validity, E3/E5).
  Until then, divergence is unexplained, not validated.
- **(B) Rankings AGREE (Spearman high).** Split by **discrimination power** (resolving power:
  spread, fewer ceiling ties, more distinct rank levels before saturation):
  - **(B1) forward resolves more finely than IGS** → IGS is the *weaker* instrument; as models
    improve it **saturates/dies first**. → **no value.**
  - **(B2) IGS resolves more finely than forward** → **best case.** Agreement gives IGS construct
    credibility (it tracks the same capability ordering), *and* its finer resolution means that as
    future models compress forward scores toward the ceiling, **IGS still discriminates** →
    durable competitive value.

Operationally: rank-agreement = Spearman(IGS, hard-forward); discrimination power = score spread /
(1 − fraction-tied-at-max) / number of distinct rank levels, measured on the same model set. This
experiment (E10, §7) is the single most decisive test of whether IGS deserves to exist.

---

## 6. Threats to validity (what a reviewer will attack — and our position)

1. **Moderate N, ties.** N up to 21 (20 for solve∩IGS) with several tied/ceilinged values →
   correlations now carry bootstrap 95% CIs (§5.4) but remain *moderately* wide; larger N would
   tighten them. → §7 (E1).
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
- **E10 — The value-decision test (most decisive, §5.6).** Run the same models on a *hard,
  non-saturated* forward benchmark (AIME-level / FrontierMath subset / hard public set).
  Compute rank-agreement Spearman(IGS, hard-forward) and each axis's discrimination power.
  Outcomes: **(A)** rankings differ → must prove IGS's construct (gate on E3/E5); **(B1)** agree
  but forward resolves finer → IGS dies first, no value; **(B2)** agree but IGS resolves finer →
  validated *and* future-proof (the win condition). This single experiment determines whether IGS
  is worth keeping.
- **E11 — Proper memorization/contamination probe (if we keep any forward claim).** Replicate the
  literature's *strong* tests rather than the weak number-swap of §5.5: (a) added-clause/distractor
  perturbation (GSM-NoOp style) on multi-step problems; (b) a word-problem domain where solution
  *paths* can be recalled; (c) a canonical-vs-fresh contamination *gap* (likely-trained items vs
  matched novel items, à la GSM1k); (d) larger N. Note: Inversa's thesis does not require this —
  it only matters if we want to assert anything about forward being contaminated.

- **E12 — Head-to-head contamination gap (Pillar-1 core proof, §0/§1).** On the same models,
  measure (a) a *forward* contamination gap = acc(likely-contaminated/standard items) −
  acc(structurally-matched fresh items), and (b) an *inverse* gap computed the same way on
  answer→problem tasks (predicted ≈ 0, since transform inputs are randomized at test time). The
  vision is supported iff **inverse gap < forward gap** with separated CIs. This is the experiment
  that proves the *original* thesis (inverse is more contamination-robust). Note: forward gaps are
  small for frontier models (GSM1k), so include weaker/mid families where the gap is large enough
  to resolve, and a contamination-prone setup (reused public items) for the forward arm.

### Related work (for citation)
- GSM-Symbolic — Mirzadeh et al., Apple, ICLR 2025 (arXiv:2410.05229): perturbation fragility;
  added clause drops frontier models up to 65%.
- GSM1k / "A Careful Examination…" — Zhang et al., Scale AI, 2024 (arXiv:2405.00332): contamination
  gap up to 13% for Phi/Mistral; minimal for GPT/Claude/Gemini.
- Rephrased-samples contamination — Yang et al., 2023 (arXiv:2311.04850): n-gram-undetectable
  contamination inflates MMLU/GSM8K/HumanEval.

---

## 8. Claims we can/can't make today

| Claim | Status |
|---|---|
| Forward benchmarks have *measurable* contamination | **Supported by literature** (GSM1k up to 13%; replica-loss; rephrase) — cite, not re-proven |
| Inverse is *recall-resistant by construction* | **Supported** (transform pinned to randomized test-time input; no fixed item to leak) |
| Inverse has a *smaller* contamination gap than forward | **NOT yet — Pillar-1 core** (needs E12 head-to-head) |
| Generative construction is *real* (not pure recall) | **Supported** (recall-proof transform spreads 0–1 via genuine substitution) |
| It is *measurable* and *internally coherent* | **Supported** (gen axes intercorrelate, CIs exclude 0; IGS defined & reproducible) |
| It discriminates where *our* solve banks saturate | **Supported, suggestive** (8 solve-100% models span IGS 0.65–1.00, N=21) |
| Forward solving here is genuine, not memorized | **NOT established** (Exp-F is only number-swaps on simple algebra; merely *agrees* with GSM1k frontier-robustness; needs E11) |
| Forward benchmarks saturate / are memorized *in general* | **NOT claimed** (contamination is real but frontier-small per GSM1k; hard benchmarks don't saturate; literature cited, not re-proven) |
| It is a *distinct dimension* from general capability | **Not yet** (ceiling confound; needs E3) |
| It adds value over a hard forward benchmark | **Untested — decisive** (needs E10 / §5.6: rank-agreement × discrimination power) |
| It *generalizes beyond algebra* | **Untested** (needs E4) |
| It *predicts anything external* | **Untested** (needs E5) |

---

## 9. Data & artifact index (for the eventual paper)

- IGS leaderboard (N=21): `data/results/igs_leaderboard.{html,json}` (built from
  `paper_level3_all.json` + `paper_experiment.json`).
- Paper batteries (N≈22): `data/results/paper_experiment.json` (solve+adversarial),
  `paper_level3_all.json` (pose+transform), `perturbation_results.json` (Experiment F).
  Earlier sweeps `macro3_*`, `macro_*`, `macro_*_v2` (solve-axis cleaning study, §5.4).
- Per-item glass-box reports: `*_report.html` (every equation, target, verdict).
- Banks: `data/banks/{struct_pose_targets,transform_bank,solve_set_graded,solve_set_hard2,
  perturbation_set}.json`.
- Methods provenance: commit history on `scoring-validity` (verifier hardening, scoring
  decomposition, ceiling-break, IGS formalization, format-artifact removal, bootstrap CIs).

---

## 10. Next action

E7 (artifacts) and E1 (scale + bootstrap CIs) are **done**. The thesis now has two falsifiable
cores, each with one decisive experiment:
- **E12 (Pillar 1, primary):** head-to-head contamination gap — does inverse leak *less* than
  forward on the same models? Proves the original "more objective benchmark" vision.
- **E10 (Pillar 2):** vs a hard non-saturated forward benchmark — rank-agreement × discrimination
  power (B2 = durable value).

Recommended: **E12 first** (it proves the primary, literature-grounded vision), then E10. After:
E3 (discriminant residual), E4 (generality), E5 (predictive). (Experiment F is retained only as a
scoped robustness note, §5.5 — it does not prove the memorization claim.)
