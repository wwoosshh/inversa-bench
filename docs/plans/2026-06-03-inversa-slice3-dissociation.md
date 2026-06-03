# Inversa Slice 3 (Dissociation Experiment + HTML Report) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Test the project's central claim (design §3): across several models, does **generative-reasoning ability (θ_gen-proxies: calibration MACE, adversarial success rate) DISSOCIATE from solving ability (θ_solve)?** If gen and solve diverge → posing measures something solving misses (pillar supported). If they track perfectly → posing is redundant (honest negative result). Also produce a **self-contained HTML report** so a human can read the results in a browser (table + bars + the dissociation verdict + honest caveats).

**Architecture:** Add a `solve` eval (θ_solve = accuracy on a fixed equation set, reusing the slice-2b solver prompt + parser). Add a per-model harness that, given adapters, computes `ModelScores(solve_accuracy, calibration_mace, adversarial_success_rate)` by reusing the existing calibration/adversarial runners. Add a tiny dependency-free `spearman` rank-correlation. Add an HTML renderer (inline CSS, percentage bars — no new deps) and an experiment CLI that runs N models and writes `report.html` + `results.json`.

**Tech Stack:** Python 3.11+, sympy, anthropic, python-dotenv, pytest (no new deps). Env at `D:\Inversa\.venv`; run via `D:/Inversa/.venv/Scripts/python.exe`; absolute paths; **commit only, no push**; branch `slice3-dissociation`.

**Scope guardrails:** gen-proxies = calibration MACE + adversarial success rate (validity is saturated → excluded). θ_solve = accuracy on a fixed solve set. Correlation is **Spearman across models** — with few models it is *suggestive, not conclusive* (state this in the report). NO IRT. NO new domains.

---

## File Structure
- Create: `src/inversa/tasks/solving.py` — solve eval (θ_solve).
- Create: `data/banks/solve_set_v1.json` — fixed solve problems (equation + answer).
- Create: `src/inversa/analysis.py` — `spearman` (dependency-free).
- Create: `src/inversa/experiment.py` — `ModelScores`, `evaluate_model`, `run_experiment`, `correlations`.
- Create: `src/inversa/report_html.py` — `render_html`.
- Create: `src/inversa/cli_experiment.py` — runs the experiment, writes HTML + JSON.
- Tests: `tests/test_solving.py`, `tests/test_analysis.py`, `tests/test_experiment.py`, `tests/test_report_html.py`.

---

### Task 0: Branch
- [ ] `git -C "D:/Inversa" checkout main` then `git -C "D:/Inversa" checkout -b slice3-dissociation`
- [ ] baseline: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests -q` → `62 passed`.

---

### Task 1: Solve eval (θ_solve) + solve set

**Files:** Create `src/inversa/tasks/solving.py`, `data/banks/solve_set_v1.json`; Test `tests/test_solving.py`.

- [ ] **Step 1: failing tests** — `tests/test_solving.py`:
```python
from inversa.adapters.fake import FakeAdapter
from inversa.tasks.solving import solve_item, solve_batch, accuracy


def test_solve_item_correct():
    a = FakeAdapter(["3"])
    it = solve_item(a, "2*x + 1 = 7", 3)
    assert it.answer == 3.0 and it.correct is True


def test_solve_item_wrong():
    a = FakeAdapter(["99"])
    it = solve_item(a, "2*x + 1 = 7", 3)
    assert it.correct is False


def test_solve_item_x_equals_reply():
    a = FakeAdapter(["x = 7"])
    it = solve_item(a, "3*x - 5 = 16", 7)
    assert it.correct is True


def test_accuracy():
    a = FakeAdapter(["3", "99", "7"])
    probs = [
        {"equation": "2*x + 1 = 7", "answer": 3},
        {"equation": "2*x + 1 = 7", "answer": 3},   # adapter returns 99 -> wrong
        {"equation": "3*x - 5 = 16", "answer": 7},
    ]
    items = solve_batch(a, probs)
    assert accuracy(items) == 2 / 3
```

- [ ] **Step 2: run → fail** (`No module named 'inversa.tasks.solving'`).

- [ ] **Step 3: implement** — `src/inversa/tasks/solving.py`:
```python
"""Forward solving eval: ask a model to solve fixed equations; theta_solve = accuracy.
Reuses the slice-2b solver prompt + numeric-answer parser."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from inversa.adapters.base import Adapter
from inversa.tasks.adversarial import build_solver_prompt, parse_numeric_answer

_TOL = 1e-6


@dataclass
class SolveItem:
    equation: str
    expected: float
    raw_output: str
    answer: Optional[float]
    correct: bool


def solve_item(adapter: Adapter, equation: str, expected) -> SolveItem:
    raw = adapter.generate(build_solver_prompt(equation))
    ans = parse_numeric_answer(raw)
    correct = ans is not None and abs(ans - float(expected)) < _TOL
    return SolveItem(equation, float(expected), raw, ans, correct)


def solve_batch(adapter: Adapter, problems) -> list:
    return [solve_item(adapter, p["equation"], p["answer"]) for p in problems]


def accuracy(items) -> float:
    n = len(items)
    return (sum(1 for it in items if it.correct) / n) if n else 0.0
```

- [ ] **Step 4: solve set** — `data/banks/solve_set_v1.json`:
```json
{
  "problems": [
    {"equation": "2*x + 1 = 7", "answer": 3},
    {"equation": "3*x - 5 = 16", "answer": 7},
    {"equation": "5*x = 60", "answer": 12},
    {"equation": "x/2 + 4 = 9", "answer": 10},
    {"equation": "2*(x + 3) = 14", "answer": 4},
    {"equation": "7*x - 2 = 4*x + 13", "answer": 5},
    {"equation": "x**3 = 27", "answer": 3},
    {"equation": "(x - 8)**3 = 0", "answer": 8}
  ]
}
```

- [ ] **Step 5: run → pass** (`4 passed`); full suite → `66 passed`. **Commit:** `git -C "D:/Inversa" add src/inversa/tasks/solving.py data/banks/solve_set_v1.json tests/test_solving.py && git -C "D:/Inversa" commit -m "feat(solving): forward solve eval (theta_solve) + solve set"`

---

### Task 2: Analysis (spearman) + experiment harness

**Files:** Create `src/inversa/analysis.py`, `src/inversa/experiment.py`; Test `tests/test_analysis.py`, `tests/test_experiment.py`.

- [ ] **Step 1: failing tests**
`tests/test_analysis.py`:
```python
from inversa.analysis import spearman


def test_perfect_positive():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_perfect_negative():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_no_variance_returns_none():
    assert spearman([1, 2, 3], [5, 5, 5]) is None


def test_too_few_points_returns_none():
    assert spearman([1], [2]) is None
```
`tests/test_experiment.py`:
```python
from inversa.adapters.fake import FakeAdapter
from inversa.experiment import ModelScores, evaluate_model, correlations


def test_evaluate_model_assembles_scores():
    # poser fake answers everything with a valid linear eq / answer; weak fake always wrong
    poser = FakeAdapter(["3", "2*x = 6", "2*x = 6"])  # solve reply, then posing replies (cycled)
    weak = FakeAdapter(["999"])
    scores = evaluate_model(
        "fake-model", poser, weak,
        solve_problems=[{"equation": "2*x + 1 = 7", "answer": 3}],
        calib_targets=[3], calib_levels=[1],
        adv_targets=[3],
    )
    assert isinstance(scores, ModelScores)
    assert scores.model == "fake-model"
    assert 0.0 <= scores.solve_accuracy <= 1.0
    assert 0.0 <= scores.adversarial_success_rate <= 1.0


def test_correlations_keys():
    scores = [
        ModelScores("a", solve_accuracy=0.9, calibration_mace=0.1, adversarial_success_rate=0.2),
        ModelScores("b", solve_accuracy=0.6, calibration_mace=0.5, adversarial_success_rate=0.5),
        ModelScores("c", solve_accuracy=0.3, calibration_mace=0.9, adversarial_success_rate=0.8),
    ]
    cors = correlations(scores)
    assert "solve_vs_calibration_mace" in cors
    assert "solve_vs_adversarial" in cors
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement**
`src/inversa/analysis.py`:
```python
"""Dependency-free Spearman rank correlation (suggestive with few points)."""
from __future__ import annotations

from typing import List, Optional, Sequence


def _avg_ranks(vals: Sequence[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1.0  # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((a[i] - ma) ** 2 for i in range(n))
    vb = sum((b[i] - mb) ** 2 for i in range(n))
    if va == 0 or vb == 0:
        return None
    return cov / ((va * vb) ** 0.5)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_avg_ranks(xs), _avg_ranks(ys))
```
`src/inversa/experiment.py`:
```python
"""Per-model evaluation + cross-model dissociation correlations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from inversa.adapters.base import Adapter
from inversa.analysis import spearman
from inversa.report import adversarial_summary, calibration_summary
from inversa.tasks.adversarial import run_adversarial_batch
from inversa.tasks.calibration import run_calibration_batch
from inversa.tasks.solving import accuracy, solve_batch


@dataclass
class ModelScores:
    model: str
    solve_accuracy: float
    calibration_mace: Optional[float]
    adversarial_success_rate: float


def evaluate_model(model: str, poser: Adapter, weak: Adapter,
                   solve_problems, calib_targets, calib_levels, adv_targets) -> ModelScores:
    solve_items = solve_batch(poser, solve_problems)
    cal = calibration_summary(run_calibration_batch(poser, calib_targets, calib_levels))
    adv = adversarial_summary(run_adversarial_batch(poser, weak, adv_targets))
    return ModelScores(
        model=model,
        solve_accuracy=accuracy(solve_items),
        calibration_mace=cal.mean_abs_calibration_error,
        adversarial_success_rate=adv.adversarial_success_rate,
    )


def correlations(scores: List[ModelScores]) -> dict:
    solve = [s.solve_accuracy for s in scores]
    adv = [s.adversarial_success_rate for s in scores]
    # calibration MACE: lower is better, so correlate solve vs MACE directly (expect
    # NEGATIVE if they track; near-zero/positive suggests dissociation). Skip models
    # whose MACE is None.
    mace_pairs = [(s.solve_accuracy, s.calibration_mace) for s in scores
                  if s.calibration_mace is not None]
    mace_corr = (spearman([p[0] for p in mace_pairs], [p[1] for p in mace_pairs])
                 if len(mace_pairs) >= 2 else None)
    return {
        "solve_vs_calibration_mace": mace_corr,
        "solve_vs_adversarial": spearman(solve, adv),
    }
```

- [ ] **Step 4: run → pass; full suite → `72 passed`.** **Commit:** `... -m "feat(experiment): per-model scores + spearman dissociation correlations"`

---

### Task 3: HTML report + experiment CLI

**Files:** Create `src/inversa/report_html.py`, `src/inversa/cli_experiment.py`; Test `tests/test_report_html.py`.

- [ ] **Step 1: failing test** — `tests/test_report_html.py`:
```python
from inversa.experiment import ModelScores
from inversa.report_html import render_html


def _scores():
    return [
        ModelScores("opus", 0.9, 0.1, 0.7),
        ModelScores("sonnet", 0.7, 0.4, 0.5),
        ModelScores("haiku", 0.4, 0.8, 0.2),
    ]


def test_html_contains_models_and_structure():
    html = render_html(_scores(), {"solve_vs_calibration_mace": -0.5, "solve_vs_adversarial": 1.0})
    assert "<html" in html.lower()
    for name in ("opus", "sonnet", "haiku"):
        assert name in html
    assert "Dissociation" in html or "dissociation" in html
    assert "%" in html  # percentage bars/labels


def test_html_handles_none_correlation():
    html = render_html(_scores(), {"solve_vs_calibration_mace": None, "solve_vs_adversarial": None})
    assert "n/a" in html.lower()
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement**
`src/inversa/report_html.py`:
```python
"""Self-contained HTML report (inline CSS, percentage bars — no external deps).
Glass-box: shows every per-model number plus the dissociation verdict and caveats."""
from __future__ import annotations

from typing import List

from inversa.experiment import ModelScores


def _bar(pct: float, color: str) -> str:
    w = max(0.0, min(100.0, pct * 100.0))
    return (f'<div style="background:#eee;border-radius:3px;width:140px;display:inline-block;'
            f'vertical-align:middle">'
            f'<div style="background:{color};width:{w:.0f}%;height:14px;border-radius:3px"></div></div>'
            f' <span>{pct:.0%}</span>')


def _fmt_corr(v) -> str:
    return "n/a" if v is None else f"{v:+.2f}"


def render_html(scores: List[ModelScores], corr: dict) -> str:
    rows = []
    for s in scores:
        mace = "n/a" if s.calibration_mace is None else f"{s.calibration_mace:.2f}"
        rows.append(
            "<tr>"
            f"<td><b>{s.model}</b></td>"
            f"<td>{_bar(s.solve_accuracy, '#2563eb')}</td>"
            f"<td style='text-align:center'>{mace}</td>"
            f"<td>{_bar(s.adversarial_success_rate, '#16a34a')}</td>"
            "</tr>"
        )
    table = "\n".join(rows)
    c_mace = _fmt_corr(corr.get("solve_vs_calibration_mace"))
    c_adv = _fmt_corr(corr.get("solve_vs_adversarial"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Inversa — dissociation report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:2rem auto;color:#1a1a1a;line-height:1.5}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border-bottom:1px solid #ddd;padding:8px 10px;text-align:left}}
th{{font-size:13px;color:#555;text-transform:uppercase;letter-spacing:.03em}}
.box{{background:#f7f7f8;border:1px solid #e5e5e5;border-radius:6px;padding:12px 16px;margin:1rem 0}}
.caveat{{color:#666;font-size:14px}}
code{{background:#eee;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>Inversa — generative vs solving (dissociation)</h1>
<p>Each model is scored on <b>solving</b> (accuracy on a fixed equation set) and on two
<b>generative</b> proxies: <b>calibration</b> (mean abs error hitting a requested difficulty degree — <i>lower is better</i>)
and <b>adversarial success</b> (rate of posing valid equations a weaker model fails). The question:
do the generative scores <b>diverge</b> from solving?</p>
<table>
<tr><th>Model</th><th>Solve accuracy</th><th>Calibration MACE ↓</th><th>Adversarial success</th></tr>
{table}
</table>
<div class="box">
<b>Dissociation (Spearman across models):</b><br>
solve vs calibration-MACE: <code>{c_mace}</code> (near 0 / positive ⇒ generative diverges from solving)<br>
solve vs adversarial: <code>{c_adv}</code> (near 0 / negative ⇒ diverges; +1 ⇒ tracks solving)
</div>
<p class="caveat"><b>Honest caveats:</b> few models ⇒ correlations are <i>suggestive, not conclusive</i>.
Difficulty = polynomial degree (a v1 proxy). All scores are machine-verified (sympy), but whether
they measure "understanding" beyond solving is exactly the hypothesis under test. Domain: math equations only.</p>
</body></html>"""
```
`src/inversa/cli_experiment.py`:
```python
"""CLI: run the dissociation experiment across models; write report.html + results.json."""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from inversa.adapters.anthropic_adapter import AnthropicAdapter
from inversa.experiment import correlations, evaluate_model
from inversa.report_html import render_html


def main(argv=None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Inversa dissociation experiment")
    parser.add_argument("--models", default="claude-opus-4-8,claude-sonnet-4-6,claude-haiku-4-5-20251001",
                        help="comma-separated poser models")
    parser.add_argument("--weak-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--solve-bank", default="data/banks/solve_set_v1.json")
    parser.add_argument("--targets", default="3,7,12,42")
    parser.add_argument("--levels", default="1,2,3")
    parser.add_argument("--out", default="data/results/report.html")
    parser.add_argument("--json-out", default="data/results/results.json")
    args = parser.parse_args(argv)

    with open(args.solve_bank, encoding="utf-8") as f:
        solve_problems = json.load(f)["problems"]
    targets = [int(x) for x in args.targets.split(",")]
    levels = [int(x) for x in args.levels.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    weak = AnthropicAdapter(model=args.weak_model)

    scores = []
    for m in models:
        poser = AnthropicAdapter(model=m)
        s = evaluate_model(m, poser, weak, solve_problems, targets, levels, targets)
        scores.append(s)
        print(f"[done] {m}: solve={s.solve_accuracy:.0%} "
              f"mace={s.calibration_mace} adv={s.adversarial_success_rate:.0%}")

    cors = correlations(scores)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(render_html(scores, cors))
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump({"scores": [s.__dict__ for s in scores], "correlations": cors}, f, indent=2)
    print(f"correlations: {cors}")
    print(f"wrote {args.out} and {args.json_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: verify (no live call)** — `pytest tests/test_report_html.py -q` → `2 passed`; `import inversa.cli_experiment` → ok; full suite → `74 passed`.
- [ ] **Step 5: commit** `... -m "feat(experiment): HTML report + dissociation experiment CLI"`
- [ ] **Step 6: live experiment (controller-run, multi-model)** — `data/results/report.html` opens in a browser. Interpretation per the report's own verdict + caveats.

---

## Self-Review
**Spec coverage:** θ_solve (T1), gen-proxies reused + per-model harness + correlations (T2), HTML report + CLI (T3). Pillar question answered by the correlations + report. ✓
**Placeholders:** none; complete code + commands. ✓
**Type consistency:** `ModelScores(model, solve_accuracy, calibration_mace, adversarial_success_rate)` used by `correlations`, `render_html`, CLI. `solve_batch`/`accuracy`/`solve_item` consistent. `spearman(xs,ys)->Optional[float]`. Reuses `build_solver_prompt`/`parse_numeric_answer` (adversarial), `run_calibration_batch`+`calibration_summary`, `run_adversarial_batch`+`adversarial_summary`. ✓
