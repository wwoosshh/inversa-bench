# PoseBench Slice 1 (Math Equation Posing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end pipeline that asks a model to *construct* a single-variable algebraic equation whose solution is a given target, machine-verifies validity with sympy, and reports a validity rate over a batch.

**Architecture:** A pure, fully-tested sympy verifier is the core. A thin `Adapter` protocol abstracts model calls (a `FakeAdapter` drives all unit tests with no network; an `AnthropicAdapter` is the real one, smoke-tested manually). A posing runner builds prompts, calls the adapter, extracts the equation, and verifies it. A report module summarizes results. A CLI runs a batch from a JSON target bank. This is the minimal vertical slice of the design doc (`docs/specs/2026-06-03-posebench-design.md` §11 step 1): posing → forward-verification → validity score. No IRT, no depth constraints, no multi-domain yet (those are later slices).

**Tech Stack:** Python 3.11+, sympy (verification), anthropic SDK (real adapter), pytest (tests), setuptools src-layout.

**Scope guardrails (design §8):** validity only (not depth — that's slice 2); math-equation sub-domain only; one provider. Do NOT add IRT, difficulty targets, or open-domain scoring here.

---

## File Structure

- Create: `pyproject.toml` — package metadata + deps, src-layout.
- Create: `src/posebench/__init__.py` — package marker.
- Create: `src/posebench/verifiers/__init__.py` — package marker.
- Create: `src/posebench/verifiers/math_equation.py` — `verify_equation(equation_str, target) -> VerificationResult`. **Core, fully unit-tested.**
- Create: `src/posebench/adapters/__init__.py` — package marker.
- Create: `src/posebench/adapters/base.py` — `Adapter` Protocol (`generate(prompt) -> str`).
- Create: `src/posebench/adapters/fake.py` — `FakeAdapter` for tests.
- Create: `src/posebench/adapters/anthropic_adapter.py` — real Claude adapter (smoke-tested).
- Create: `src/posebench/tasks/__init__.py` — package marker.
- Create: `src/posebench/tasks/posing.py` — `build_prompt`, `_extract_equation`, `ItemResult`, `run_item`, `run_batch`.
- Create: `src/posebench/report.py` — `Summary`, `summarize`, `format_summary`.
- Create: `src/posebench/cli.py` — argparse entry point.
- Create: `data/banks/math_targets_slice1.json` — small target bank.
- Create: `tests/test_math_equation_verifier.py`
- Create: `tests/test_fake_adapter.py`
- Create: `tests/test_posing_runner.py`
- Create: `tests/test_report.py`

Each file has one responsibility; files that change together (verifier + its test) are introduced together.

---

### Task 0: Project setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/posebench/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "posebench"
version = "0.1.0"
description = "Measure AI reasoning by problem-posing with machine-verifiable scoring"
requires-python = ">=3.11"
dependencies = [
    "sympy>=1.12",
    "anthropic>=0.40",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package markers**

`src/posebench/__init__.py`:
```python
"""PoseBench: measure AI reasoning by problem-posing."""
__version__ = "0.1.0"
```

`tests/__init__.py`:
```python
```
(empty file)

- [ ] **Step 3: Create venv and install (editable + dev deps)**

Run (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```
Expected: installs posebench (editable), sympy, anthropic, pytest with no errors.

- [ ] **Step 4: Verify pytest collects (0 tests yet)**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: "no tests ran" (exit code 5) — confirms pytest + package import work.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/posebench/__init__.py tests/__init__.py
git commit -m "chore: project setup (pyproject, package skeleton, venv install)"
```

---

### Task 1: Math equation verifier (core)

**Files:**
- Create: `src/posebench/verifiers/__init__.py` (empty)
- Create: `src/posebench/verifiers/math_equation.py`
- Test: `tests/test_math_equation_verifier.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_math_equation_verifier.py`:
```python
from posebench.verifiers.math_equation import verify_equation


def test_linear_equation_valid_and_unique():
    r = verify_equation("2*x + 1 = 7", 3)
    assert r.well_formed is True
    assert r.valid is True
    assert r.unique is True
    assert r.error is None


def test_implicit_multiplication_parsed():
    r = verify_equation("2x = 6", 3)
    assert r.well_formed is True
    assert r.valid is True
    assert r.unique is True


def test_quadratic_valid_but_not_unique():
    r = verify_equation("x**2 = 9", 3)
    assert r.well_formed is True
    assert r.valid is True
    assert r.unique is False  # solutions are 3 and -3


def test_target_zero():
    r = verify_equation("x = 0", 0)
    assert r.valid is True and r.unique is True


def test_wrong_target_is_invalid():
    r = verify_equation("2*x = 6", 5)
    assert r.well_formed is True
    assert r.valid is False


def test_no_variable_is_not_well_formed():
    r = verify_equation("2 + 2 = 4", 4)
    assert r.well_formed is False
    assert r.valid is False
    assert "x" in r.error


def test_missing_equals_is_not_well_formed():
    r = verify_equation("2*x + 1", 3)
    assert r.well_formed is False
    assert r.valid is False


def test_parse_error_is_not_well_formed():
    r = verify_equation("x + = 3", 3)
    assert r.well_formed is False
    assert r.valid is False
    assert r.error is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_math_equation_verifier.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'posebench.verifiers'`.

- [ ] **Step 3: Implement the verifier**

`src/posebench/verifiers/__init__.py`:
```python
```
(empty file)

`src/posebench/verifiers/math_equation.py`:
```python
"""Forward-verification oracle for posed single-variable equations.

Given a target value, check whether a model-posed equation in `x` is
well-formed and actually has that target among its real solutions.
This is the slice-1 "validity" layer (depth constraints come later).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_X = sp.Symbol("x")
_TOL = 1e-6


@dataclass
class VerificationResult:
    well_formed: bool        # parses as an equation in x
    valid: bool              # target is among the real solutions
    unique: bool             # the real solution set is exactly {target}
    solutions: List[str]     # string repr of all solutions (evidence)
    error: Optional[str] = None


def _real_solutions(solutions) -> List[float]:
    reals: List[float] = []
    for sol in solutions:
        try:
            val = complex(sol)
        except (TypeError, ValueError):
            continue  # symbolic / non-numeric solution -> skip
        if abs(val.imag) < 1e-9:
            reals.append(val.real)
    return reals


def verify_equation(equation_str: str, target: float) -> VerificationResult:
    s = (equation_str or "").strip()
    if s.count("=") != 1:
        return VerificationResult(False, False, False, [],
                                  "equation must contain exactly one '='")
    lhs_str, rhs_str = s.split("=")
    try:
        lhs = parse_expr(lhs_str, transformations=_TRANSFORMS,
                         local_dict={"x": _X}, evaluate=True)
        rhs = parse_expr(rhs_str, transformations=_TRANSFORMS,
                         local_dict={"x": _X}, evaluate=True)
    except Exception as e:  # parse failure -> not well formed
        return VerificationResult(False, False, False, [], f"parse error: {e}")

    if _X not in (lhs - rhs).free_symbols:
        return VerificationResult(False, False, False, [],
                                  "equation does not involve variable x")

    eq = sp.Eq(lhs, rhs)
    try:
        solutions = sp.solve(eq, _X, dict=False)
    except Exception as e:
        return VerificationResult(True, False, False, [], f"solve error: {e}")

    reals = _real_solutions(solutions)
    target_f = float(target)
    valid = any(abs(rs - target_f) < _TOL for rs in reals)
    unique = valid and len(reals) == 1
    return VerificationResult(True, valid, unique, [str(sol) for sol in solutions])
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_math_equation_verifier.py -q
```
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/posebench/verifiers/ tests/test_math_equation_verifier.py
git commit -m "feat(verifier): sympy forward-verification of posed equations (validity)"
```

---

### Task 2: Adapter protocol + FakeAdapter

**Files:**
- Create: `src/posebench/adapters/__init__.py` (empty)
- Create: `src/posebench/adapters/base.py`
- Create: `src/posebench/adapters/fake.py`
- Test: `tests/test_fake_adapter.py`

- [ ] **Step 1: Write the failing test**

`tests/test_fake_adapter.py`:
```python
from posebench.adapters.fake import FakeAdapter


def test_fake_adapter_returns_scripted_outputs_in_order():
    a = FakeAdapter(["2*x = 6", "x = 1"])
    assert a.generate("ignored prompt") == "2*x = 6"
    assert a.generate("ignored prompt") == "x = 1"


def test_fake_adapter_cycles_when_exhausted():
    a = FakeAdapter(["only"])
    assert a.generate("p") == "only"
    assert a.generate("p") == "only"


def test_fake_adapter_records_prompts():
    a = FakeAdapter(["x = 0"])
    a.generate("first prompt")
    assert a.prompts == ["first prompt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_fake_adapter.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'posebench.adapters'`.

- [ ] **Step 3: Implement adapter base + fake**

`src/posebench/adapters/__init__.py`:
```python
```
(empty file)

`src/posebench/adapters/base.py`:
```python
"""Adapter protocol: anything that turns a prompt into a text completion."""
from __future__ import annotations

from typing import Protocol


class Adapter(Protocol):
    def generate(self, prompt: str) -> str:
        """Return the model's text completion for `prompt`."""
        ...
```

`src/posebench/adapters/fake.py`:
```python
"""Deterministic fake adapter for tests (no network)."""
from __future__ import annotations

from typing import List, Sequence


class FakeAdapter:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses: List[str] = list(responses)
        self._i = 0
        self.prompts: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise ValueError("FakeAdapter has no responses configured")
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_fake_adapter.py -q
```
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/posebench/adapters/__init__.py src/posebench/adapters/base.py src/posebench/adapters/fake.py tests/test_fake_adapter.py
git commit -m "feat(adapters): Adapter protocol + FakeAdapter for tests"
```

---

### Task 3: Posing runner

**Files:**
- Create: `src/posebench/tasks/__init__.py` (empty)
- Create: `src/posebench/tasks/posing.py`
- Test: `tests/test_posing_runner.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_posing_runner.py`:
```python
from posebench.adapters.fake import FakeAdapter
from posebench.tasks.posing import build_prompt, _extract_equation, run_item, run_batch


def test_build_prompt_mentions_target_and_x():
    p = build_prompt(3)
    assert "3" in p
    assert "x" in p


def test_extract_equation_takes_first_line_with_equals():
    raw = "Here you go:\n2*x + 1 = 7\nthanks"
    assert _extract_equation(raw) == "2*x + 1 = 7"


def test_extract_equation_strips_backticks():
    assert _extract_equation("`x = 5`") == "x = 5"


def test_run_item_valid_equation():
    adapter = FakeAdapter(["2*x = 6"])
    result = run_item(adapter, 3)
    assert result.target == 3.0
    assert result.equation == "2*x = 6"
    assert result.verification.valid is True


def test_run_item_invalid_equation():
    adapter = FakeAdapter(["2*x = 6"])
    result = run_item(adapter, 5)  # 2x=6 solves to 3, not 5
    assert result.verification.valid is False


def test_run_batch_returns_one_result_per_target():
    adapter = FakeAdapter(["x = 1", "x = 2", "x = 3"])
    results = run_batch(adapter, [1, 2, 3])
    assert len(results) == 3
    assert all(r.verification.valid for r in results)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_posing_runner.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'posebench.tasks'`.

- [ ] **Step 3: Implement the runner**

`src/posebench/tasks/__init__.py`:
```python
```
(empty file)

`src/posebench/tasks/posing.py`:
```python
"""Inverse/posing task: target answer -> model constructs an equation -> verify."""
from __future__ import annotations

from dataclasses import dataclass

from posebench.adapters.base import Adapter
from posebench.verifiers.math_equation import verify_equation, VerificationResult

PROMPT_TEMPLATE = (
    "You are constructing a math problem.\n"
    "Write a single-variable algebraic EQUATION in the variable x whose unique "
    "solution is exactly {target}.\n"
    "Output ONLY the equation in the form '<lhs> = <rhs>', with no explanation.\n"
    "Use Python/sympy syntax (examples: 2*x + 1 = 7 ; x**2 - 9 = 0).\n"
)


def build_prompt(target) -> str:
    return PROMPT_TEMPLATE.format(target=target)


def _extract_equation(raw: str) -> str:
    for line in raw.splitlines():
        if "=" in line:
            return line.strip().strip("`").strip()
    return raw.strip().strip("`").strip()


@dataclass
class ItemResult:
    target: float
    raw_output: str
    equation: str
    verification: VerificationResult


def run_item(adapter: Adapter, target) -> ItemResult:
    raw = adapter.generate(build_prompt(target))
    equation = _extract_equation(raw)
    verification = verify_equation(equation, target)
    return ItemResult(float(target), raw, equation, verification)


def run_batch(adapter: Adapter, targets) -> list:
    return [run_item(adapter, t) for t in targets]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_posing_runner.py -q
```
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/posebench/tasks/ tests/test_posing_runner.py
git commit -m "feat(tasks): posing runner (prompt -> extract -> verify)"
```

---

### Task 4: Report

**Files:**
- Create: `src/posebench/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:
```python
from posebench.adapters.fake import FakeAdapter
from posebench.tasks.posing import run_batch
from posebench.report import summarize, format_summary


def _results():
    # two valid (x=1, x=2), one invalid for target 9 (adapter returns x=1)
    adapter = FakeAdapter(["x = 1", "x = 2", "x = 1"])
    return run_batch(adapter, [1, 2, 9])


def test_summarize_counts():
    s = summarize(_results())
    assert s.n == 3
    assert s.n_valid == 2
    assert s.validity_rate == 2 / 3


def test_summarize_empty():
    s = summarize([])
    assert s.n == 0
    assert s.validity_rate == 0.0


def test_format_summary_is_readable():
    s = summarize(_results())
    text = format_summary(s)
    assert "valid" in text
    assert "3" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_report.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'posebench.report'`.

- [ ] **Step 3: Implement the report module**

`src/posebench/report.py`:
```python
"""Summarize a batch of posing results into validity / uniqueness rates."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Summary:
    n: int
    n_valid: int
    n_unique: int
    validity_rate: float
    uniqueness_rate: float


def summarize(results) -> Summary:
    n = len(results)
    n_valid = sum(1 for r in results if r.verification.valid)
    n_unique = sum(1 for r in results if r.verification.unique)
    return Summary(
        n=n,
        n_valid=n_valid,
        n_unique=n_unique,
        validity_rate=(n_valid / n if n else 0.0),
        uniqueness_rate=(n_unique / n if n else 0.0),
    )


def format_summary(s: Summary) -> str:
    return (
        f"items={s.n} valid={s.n_valid} ({s.validity_rate:.1%}) "
        f"unique={s.n_unique} ({s.uniqueness_rate:.1%})"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_report.py -q
```
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full suite + commit**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: PASS (20 passed).

```bash
git add src/posebench/report.py tests/test_report.py
git commit -m "feat(report): validity/uniqueness rate summary"
```

---

### Task 5: Anthropic adapter + CLI (real, smoke-tested)

**Files:**
- Create: `src/posebench/adapters/anthropic_adapter.py`
- Create: `src/posebench/cli.py`

> No unit tests here (these hit the network / parse argv). Verified by the end-to-end smoke run in Task 6. Keep them thin so the untested surface is minimal.

- [ ] **Step 1: Implement the Anthropic adapter**

`src/posebench/adapters/anthropic_adapter.py`:
```python
"""Real model adapter backed by the Anthropic API."""
from __future__ import annotations

import os

from anthropic import Anthropic


class AnthropicAdapter:
    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None,
                 max_tokens: int = 256) -> None:
        self._client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")
```

- [ ] **Step 2: Implement the CLI**

`src/posebench/cli.py`:
```python
"""Run a posing batch from a JSON target bank and print a report."""
from __future__ import annotations

import argparse
import json

from posebench.adapters.anthropic_adapter import AnthropicAdapter
from posebench.report import format_summary, summarize
from posebench.tasks.posing import run_batch


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="PoseBench slice-1 math runner")
    parser.add_argument("--bank", required=True, help="path to target bank JSON")
    parser.add_argument("--model", default="claude-opus-4-8")
    args = parser.parse_args(argv)

    with open(args.bank, encoding="utf-8") as f:
        targets = json.load(f)["targets"]

    adapter = AnthropicAdapter(model=args.model)
    results = run_batch(adapter, targets)

    print(format_summary(summarize(results)))
    for r in results:
        v = r.verification
        print(f"  target={r.target} eq={r.equation!r} "
              f"valid={v.valid} unique={v.unique}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Confirm imports resolve (no run)**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import posebench.cli, posebench.adapters.anthropic_adapter; print('ok')"
```
Expected: prints `ok` (no import errors).

- [ ] **Step 4: Commit**

```bash
git add src/posebench/adapters/anthropic_adapter.py src/posebench/cli.py
git commit -m "feat(cli): Anthropic adapter + batch runner CLI"
```

---

### Task 6: Target bank + end-to-end smoke run (slice milestone)

**Files:**
- Create: `data/banks/math_targets_slice1.json`

- [ ] **Step 1: Create the target bank**

`data/banks/math_targets_slice1.json`:
```json
{
  "domain": "math_equation",
  "targets": [3, 7, 12, 0, -5, 100, 0.5, 42]
}
```

- [ ] **Step 2: Run the full unit suite (must stay green)**

Run:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: PASS (20 passed).

- [ ] **Step 3: End-to-end smoke run against the real model**

Requires `ANTHROPIC_API_KEY` in the environment.

Run:
```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
.\.venv\Scripts\python.exe -m posebench.cli --bank data\banks\math_targets_slice1.json
```
Expected: a summary line like `items=8 valid=8 (100.0%) unique=...` followed by one line per target showing the posed equation and its verdict. (Validity should be high — posing an equation for a known answer is easy; this confirms the *pipeline* works end-to-end. Low-validity or extraction failures here are findings to feed into slice 2.)

- [ ] **Step 4: Commit**

```bash
git add data/banks/math_targets_slice1.json
git commit -m "feat: slice-1 math target bank + end-to-end smoke run"
```

---

## What slice 1 deliberately does NOT do (hand-off to later slices)

- **No depth measurement** — validity only. A model can pose trivial equations; that is expected. Slice 2 adds machine-checkable depth constraints (difficulty targets, required techniques, adversarial posing) per design §5.4.
- **No IRT / θ estimation** — slice 3 (design §5.5).
- **No θ_solve comparison** — slice 4 (the falsifiable dissociation experiment, design §3).
- **No code/logic domains** — slice 5.

These are out of scope here by design; do not add them to this plan.

---

## Self-Review

**1. Spec coverage (design §11 step 1 = "math domain: answer→problem posing + sympy forward-verification + validity score, one model end-to-end"):**
- answer→problem posing → Task 3 (runner) + Task 5 (real adapter). ✓
- sympy forward-verification → Task 1 (verifier). ✓
- validity score → Task 4 (report). ✓
- one model end-to-end → Task 5 + Task 6 (smoke run). ✓
- Depth/IRT/θ_solve correctly deferred (design marks them as later slices). ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step has complete code; every command has expected output. ✓

**3. Type consistency:** `VerificationResult` fields (`well_formed`, `valid`, `unique`, `solutions`, `error`) used identically in verifier (Task 1), runner asserts (Task 3), and report (`r.verification.valid`/`.unique`, Task 4). Function names consistent across tasks: `verify_equation`, `FakeAdapter`, `build_prompt`, `_extract_equation`, `run_item`, `run_batch`, `ItemResult`, `summarize`, `format_summary`, `Summary`, `AnthropicAdapter`. Test count: 8 (verifier) + 3 (fake) + 6 (runner) + 3 (report) = 20, matching the "20 passed" expectations in Tasks 4 & 6. ✓
