# Inversa Slice 2b (Adversarial Posing, model-vs-model) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Measure whether a *poser* model can construct a VALID equation (sympy confirms target is a real solution) that a designated *weaker solver model* gets WRONG. Adversarial success = `valid AND weak-model-solved-wrong`. The per-poser adversarial success rate is a gen-proxy **decoupled from the poser's own solving** (the design's elevated gen/solve separator, §5.4).

**Architecture:** New `tasks/adversarial.py`: prompts (poser "make it hard"; solver "give the number"), a robust `parse_numeric_answer`, and a two-adapter runner (`run_adversarial_item/batch`) that poses → verifies validity (sympy, strong) → asks the weak model to solve → compares. Report adds adversarial-success-rate + weak-solver-accuracy. New `cli_adversarial.py` wires poser + weak models (both via the same Anthropic key, different model ids). Reuse `verify_equation`, `extract_equation`, `AnthropicAdapter`.

**Tech Stack:** Python 3.11+, sympy, anthropic, python-dotenv, pytest. Env at `D:\Inversa\.venv`; run via `D:/Inversa/.venv/Scripts/python.exe`; absolute paths; **commit only, no push**; branch `slice2b-adversarial`.

**Scope guardrails:** model-vs-model adversarial ONLY. `valid` = target is among real solutions (reuse verify_equation; uniqueness not required here). NO IRT (slice 4), NO cross-model dissociation analysis yet (slice 3). Unit tests use FakeAdapters (no network); live two-model smoke is controller-run.

---

## File Structure
- Create: `src/inversa/tasks/adversarial.py`
- Modify: `src/inversa/report.py` (append adversarial summary)
- Create: `src/inversa/cli_adversarial.py`
- Tests: `tests/test_adversarial.py`, `tests/test_report_adversarial.py`, `tests/test_cli_adversarial.py`
- Reuse bank `data/banks/math_targets_slice1.json`.

---

### Task 0: Branch
- [ ] **Step 1**
```bash
git -C "D:/Inversa" checkout main
git -C "D:/Inversa" checkout -b slice2b-adversarial
```
- [ ] **Step 2: baseline** `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests -q` → expect `44 passed`.

---

### Task 1: Adversarial runner + numeric-answer parser

**Files:** Create `src/inversa/tasks/adversarial.py`; Test `tests/test_adversarial.py`.

- [ ] **Step 1: Write the failing tests**
`tests/test_adversarial.py`:
```python
from inversa.adapters.fake import FakeAdapter
from inversa.tasks.adversarial import (
    build_poser_prompt,
    build_solver_prompt,
    parse_numeric_answer,
    run_adversarial_item,
    run_adversarial_batch,
)


def test_poser_prompt_mentions_target_and_x():
    p = build_poser_prompt(3)
    assert "3" in p and "x" in p


def test_solver_prompt_contains_equation():
    assert "2*x = 6" in build_solver_prompt("2*x = 6")


def test_parse_plain_integer():
    assert parse_numeric_answer("3") == 3.0


def test_parse_x_equals_form():
    assert parse_numeric_answer("x = 7") == 7.0
    assert parse_numeric_answer("x=-5") == -5.0


def test_parse_fraction():
    assert parse_numeric_answer("1/2") == 0.5


def test_parse_embedded_in_prose():
    assert parse_numeric_answer("The solution is 42") == 42.0


def test_parse_no_number_returns_none():
    assert parse_numeric_answer("no real solution") is None


def test_adversarial_success_when_weak_wrong():
    poser = FakeAdapter(["(x - 3)*(x**2 + 1) = 0"])  # valid for target 3, real root only 3
    weak = FakeAdapter(["5"])                         # weak model answers wrong
    item = run_adversarial_item(poser, weak, 3)
    assert item.valid is True
    assert item.weak_answer == 5.0
    assert item.weak_correct is False
    assert item.adversarial_success is True


def test_no_adversarial_when_weak_correct():
    poser = FakeAdapter(["(x - 3)*(x**2 + 1) = 0"])
    weak = FakeAdapter(["x = 3"])
    item = run_adversarial_item(poser, weak, 3)
    assert item.valid is True
    assert item.weak_correct is True
    assert item.adversarial_success is False


def test_invalid_pose_is_never_adversarial_success():
    poser = FakeAdapter(["2*x = 6"])  # solves to 3, not 5 -> invalid for target 5
    weak = FakeAdapter(["999"])
    item = run_adversarial_item(poser, weak, 5)
    assert item.valid is False
    assert item.adversarial_success is False


def test_batch_size():
    poser = FakeAdapter(["x = 1"])
    weak = FakeAdapter(["0"])
    items = run_adversarial_batch(poser, weak, [1, 2, 3])
    assert len(items) == 3
```

- [ ] **Step 2: Run to verify fail**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_adversarial.py -q` → FAIL (`No module named 'inversa.tasks.adversarial'`).

- [ ] **Step 3: Implement**
`src/inversa/tasks/adversarial.py`:
```python
"""Adversarial posing (model-vs-model): the poser must construct a VALID equation
(target is a real solution, confirmed by sympy) that a weaker SOLVER model gets
WRONG. adversarial_success = valid AND weak model solved it incorrectly. This
measures the poser's model-of-difficulty, decoupled from its own solving."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import sympy as sp

from inversa.adapters.base import Adapter
from inversa.tasks.posing import extract_equation
from inversa.verifiers.math_equation import verify_equation, VerificationResult

POSER_PROMPT = (
    "You are constructing a HARD math problem.\n"
    "Write a single-variable algebraic EQUATION in the variable x whose UNIQUE real "
    "solution is exactly {target}, but make it tricky enough that a LESS capable solver "
    "would get it wrong.\n"
    "Output ONLY the equation as '<lhs> = <rhs>', with no explanation. Use sympy syntax."
)

SOLVER_PROMPT = (
    "Solve this equation for x:\n{equation}\n"
    "Output ONLY the numeric value of x (a single number), nothing else."
)

_TOL = 1e-6
_NUM_RE = re.compile(r"-?\d+/\d+|-?\d+\.?\d*")


def build_poser_prompt(target) -> str:
    return POSER_PROMPT.format(target=target)


def build_solver_prompt(equation: str) -> str:
    return SOLVER_PROMPT.format(equation=equation)


def parse_numeric_answer(text: str) -> Optional[float]:
    """Best-effort extraction of a single numeric answer from a model's solve reply.
    Handles '3', 'x = 7', 'x=-5', '1/2', and a number embedded in prose. Returns
    None when no number is present."""
    s = (text or "").strip()
    s = re.sub(r"^\s*x\s*=\s*", "", s, flags=re.IGNORECASE)
    candidates = [s] + list(reversed(_NUM_RE.findall(s)))
    for cand in candidates:
        try:
            return float(sp.sympify(cand))
        except Exception:
            continue
    return None


@dataclass
class AdversarialItem:
    target: float
    raw_poser_output: str
    equation: str
    verification: VerificationResult
    valid: bool                  # target is a real solution (sympy)
    weak_raw_output: str
    weak_answer: Optional[float]
    weak_correct: bool
    adversarial_success: bool    # valid AND weak model got it wrong


def run_adversarial_item(poser: Adapter, weak_solver: Adapter, target) -> AdversarialItem:
    raw = poser.generate(build_poser_prompt(target))
    equation = extract_equation(raw)
    ver = verify_equation(equation, target)
    weak_raw = weak_solver.generate(build_solver_prompt(equation))
    weak_ans = parse_numeric_answer(weak_raw)
    weak_correct = weak_ans is not None and abs(weak_ans - float(target)) < _TOL
    return AdversarialItem(
        target=float(target),
        raw_poser_output=raw,
        equation=equation,
        verification=ver,
        valid=ver.valid,
        weak_raw_output=weak_raw,
        weak_answer=weak_ans,
        weak_correct=weak_correct,
        adversarial_success=(ver.valid and not weak_correct),
    )


def run_adversarial_batch(poser: Adapter, weak_solver: Adapter, targets) -> list:
    return [run_adversarial_item(poser, weak_solver, t) for t in targets]
```

- [ ] **Step 4: Run to verify pass**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_adversarial.py -q` → expect `11 passed`.
Then full suite → expect `55 passed`.

- [ ] **Step 5: Commit**
```bash
git -C "D:/Inversa" add src/inversa/tasks/adversarial.py tests/test_adversarial.py
git -C "D:/Inversa" commit -m "feat(adversarial): model-vs-model adversarial posing runner + numeric-answer parser"
```

---

### Task 2: Adversarial report

**Files:** Modify `src/inversa/report.py` (append); Test `tests/test_report_adversarial.py`.

- [ ] **Step 1: Write the failing tests**
`tests/test_report_adversarial.py`:
```python
from inversa.adapters.fake import FakeAdapter
from inversa.tasks.adversarial import run_adversarial_batch
from inversa.report import adversarial_summary, format_adversarial_summary


def test_summary_counts():
    # both poses valid (target 3); weak wrong then correct -> 1 adversarial success of 2
    poser = FakeAdapter(["(x - 3)*(x**2 + 1) = 0"])
    weak = FakeAdapter(["5", "3"])
    items = run_adversarial_batch(poser, weak, [3, 3])
    s = adversarial_summary(items)
    assert s.n == 2
    assert s.n_valid == 2
    assert s.n_adversarial_success == 1
    assert s.adversarial_success_rate == 0.5
    assert s.weak_solver_accuracy_on_valid == 0.5


def test_summary_empty():
    s = adversarial_summary([])
    assert s.n == 0
    assert s.adversarial_success_rate == 0.0
    assert s.weak_solver_accuracy_on_valid is None


def test_format_is_readable():
    poser = FakeAdapter(["(x - 3)*(x**2 + 1) = 0"])
    weak = FakeAdapter(["5", "3"])
    items = run_adversarial_batch(poser, weak, [3, 3])
    text = format_adversarial_summary(adversarial_summary(items))
    assert "adversarial" in text.lower()
```

- [ ] **Step 2: Run to verify fail**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_report_adversarial.py -q` → FAIL (`cannot import name 'adversarial_summary'`).

- [ ] **Step 3: Implement — APPEND to `src/inversa/report.py`** (keep all existing code):
```python
@dataclass
class AdversarialSummary:
    n: int
    n_valid: int
    validity_rate: float
    n_adversarial_success: int           # valid AND weak model wrong
    adversarial_success_rate: float      # over all n
    weak_solver_accuracy_on_valid: float | None   # weak correct / valid posed equations


def adversarial_summary(items) -> AdversarialSummary:
    n = len(items)
    n_valid = sum(1 for it in items if it.valid)
    n_adv = sum(1 for it in items if it.adversarial_success)
    valid_items = [it for it in items if it.valid]
    weak_acc = (sum(1 for it in valid_items if it.weak_correct) / len(valid_items)
                if valid_items else None)
    return AdversarialSummary(
        n=n,
        n_valid=n_valid,
        validity_rate=(n_valid / n if n else 0.0),
        n_adversarial_success=n_adv,
        adversarial_success_rate=(n_adv / n if n else 0.0),
        weak_solver_accuracy_on_valid=weak_acc,
    )


def format_adversarial_summary(s: AdversarialSummary) -> str:
    wacc = "n/a" if s.weak_solver_accuracy_on_valid is None else f"{s.weak_solver_accuracy_on_valid:.1%}"
    return (f"items={s.n} valid={s.n_valid} ({s.validity_rate:.1%}) "
            f"adversarial_success={s.n_adversarial_success} ({s.adversarial_success_rate:.1%}) "
            f"weak_solver_accuracy_on_valid={wacc}")
```

- [ ] **Step 4: Run to verify pass + full suite**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_report_adversarial.py -q` → expect `3 passed`.
Run: full suite → expect `58 passed`.

- [ ] **Step 5: Commit**
```bash
git -C "D:/Inversa" add src/inversa/report.py tests/test_report_adversarial.py
git -C "D:/Inversa" commit -m "feat(report): adversarial summary (success rate + weak-solver accuracy)"
```

---

### Task 3: Adversarial CLI

**Files:** Create `src/inversa/cli_adversarial.py`; Test `tests/test_cli_adversarial.py`. (Live two-model smoke is controller-run, not here.)

- [ ] **Step 1: Write the failing test** (validation path, no network)
`tests/test_cli_adversarial.py`:
```python
import json

import pytest

from inversa.cli_adversarial import main


def test_rejects_bank_without_targets(tmp_path):
    bank = tmp_path / "b.json"
    bank.write_text(json.dumps({"domain": "math_equation"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--bank", str(bank)])
```

- [ ] **Step 2: Run to verify fail**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_cli_adversarial.py -q` → FAIL (`No module named 'inversa.cli_adversarial'`).

- [ ] **Step 3: Implement**
`src/inversa/cli_adversarial.py`:
```python
"""CLI: run a model-vs-model adversarial posing batch and print a report."""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from inversa.adapters.anthropic_adapter import AnthropicAdapter
from inversa.report import adversarial_summary, format_adversarial_summary
from inversa.tasks.adversarial import run_adversarial_batch


def main(argv=None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Inversa adversarial (model-vs-model) runner")
    parser.add_argument("--bank", required=True, help="path to target bank JSON")
    parser.add_argument("--poser-model", default="claude-opus-4-8")
    parser.add_argument("--weak-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--verbose", action="store_true",
                        help="also print each item's raw poser/weak output")
    args = parser.parse_args(argv)

    with open(args.bank, encoding="utf-8") as f:
        data = json.load(f)
    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        parser.error(f"bank {args.bank!r} must contain a non-empty 'targets' list")

    poser = AnthropicAdapter(model=args.poser_model)
    weak = AnthropicAdapter(model=args.weak_model)
    items = run_adversarial_batch(poser, weak, targets)

    # Glass-box output (design §3.5): per-item equation, validity, weak answer, verdict.
    print(format_adversarial_summary(adversarial_summary(items)))
    for it in items:
        print(f"  target={it.target} eq={it.equation!r} valid={it.valid} "
              f"weak_answer={it.weak_answer} weak_correct={it.weak_correct} "
              f"adversarial_success={it.adversarial_success}")
        if args.verbose:
            print(f"    poser_raw={it.raw_poser_output!r}")
            print(f"    weak_raw={it.weak_raw_output!r}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify (no live call)**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_cli_adversarial.py -q` → expect `1 passed`.
Run: `D:/Inversa/.venv/Scripts/python.exe -c "import inversa.cli_adversarial; print('ok')"` → `ok`.
Run: full suite → expect `59 passed`.

- [ ] **Step 5: Commit**
```bash
git -C "D:/Inversa" add src/inversa/cli_adversarial.py tests/test_cli_adversarial.py
git -C "D:/Inversa" commit -m "feat(cli): adversarial (model-vs-model) runner CLI"
```

- [ ] **Step 6: Live smoke (controller-run, two real models)**
```
D:\Inversa\.venv\Scripts\python.exe -m inversa.cli_adversarial --bank <small-bank> --verbose
```
Expected: summary `items=.. valid=.. adversarial_success=.. (..%) weak_solver_accuracy_on_valid=..%` + per-item poser eq / weak answer / verdict. **Interpretation:** high adversarial_success ⇒ poser can craft valid equations the weak model flubs (strong model-of-difficulty, decoupled from poser's own solving). weak_solver_accuracy shows how often the weak model still cracked them. Feeds slice 3 (θ_gen-proxy vs θ_solve across models).

---

## Self-Review
**Spec coverage:** runner + numeric parser (Task 1) ✓; report success-rate/weak-accuracy (Task 2) ✓; CLI + glass-box + smoke (Task 3) ✓. IRT/cross-model-analysis deferred. ✓
**Placeholder scan:** none; complete code + commands everywhere (Step 6 bank is a controller choice, intentionally `<small-bank>`). ✓
**Type consistency:** `AdversarialItem(target, raw_poser_output, equation, verification, valid, weak_raw_output, weak_answer, weak_correct, adversarial_success)` — fields consumed by report (`it.valid`, `it.adversarial_success`, `it.weak_correct`) and cli (`it.equation`, `it.weak_answer`, ...). `parse_numeric_answer -> Optional[float]`. `adversarial_summary` fields match `format_adversarial_summary` + tests. Reuses public `extract_equation` (slice-2b depends on the slice-2 rename). ✓
