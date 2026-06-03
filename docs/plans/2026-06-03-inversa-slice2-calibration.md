# Inversa Slice 2 (Difficulty-Target Calibration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Measure whether a model can pose a valid equation (solves to target N) AT A REQUESTED polynomial degree D, and score its *calibration* (|requested − actual degree|) — a metacognitive "understands difficulty" signal distinct from solving.

**Architecture:** Reuse slice-1 pieces. Add (1) a glass-box `difficulty()` proxy = polynomial degree of (lhs−rhs) in x, clamped to band 1..5 (non-polynomial → 5); (2) a calibration runner that asks the model to pose at a target degree, then measures actual degree + validity + calibration error; (3) a calibration report (validity rate + mean absolute calibration error over valid items); (4) a calibration CLI. Difficulty axis v1 = **degree only** (design §12 notes richer metrics are future work). Decision recorded per design §5.4 (difficulty-target posing).

**Tech Stack:** Python 3.11+, sympy, anthropic, python-dotenv, pytest. Env already set up at `D:\Inversa\.venv`; run via `D:/Inversa/.venv/Scripts/python.exe`; absolute paths; **commit only, no push**; work on branch `slice2-calibration` (created by Task 0).

**Scope guardrails:** difficulty axis = polynomial degree ONLY (no op-count/solution-class/step-count — future work). Calibration error measured over VALID items only. NO adversarial weak-solver yet (that is slice 2b), NO IRT (slice 4). Do not add them here.

---

## File Structure
- Modify: `src/inversa/verifiers/math_equation.py` — extract public `parse_sides()` (DRY; verify_equation reuses it). Behavior-preserving.
- Create: `src/inversa/verifiers/difficulty.py` — `difficulty(equation_str) -> DifficultyResult`.
- Create: `src/inversa/tasks/calibration.py` — prompt + `run_calibration_item` / `run_calibration_batch` + `CalibrationItem`.
- Modify: `src/inversa/report.py` — add `CalibrationSummary`, `calibration_summary`, `format_calibration_summary`.
- Create: `src/inversa/cli_calibration.py` — calibration CLI.
- Tests: `tests/test_difficulty.py`, `tests/test_calibration.py`, `tests/test_report_calibration.py`, `tests/test_cli_calibration.py`.
- Reuse existing bank `data/banks/math_targets_slice1.json`.

---

### Task 0: Branch

- [ ] **Step 1: Create branch**
```bash
git -C "D:/Inversa" checkout main
git -C "D:/Inversa" checkout -b slice2-calibration
```
- [ ] **Step 2: Confirm clean baseline**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests -q` → expect `25 passed`.

---

### Task 1: Extract `parse_sides()` (DRY refactor, behavior-preserving)

**Files:** Modify `src/inversa/verifiers/math_equation.py`. Existing tests (`tests/test_math_equation_verifier.py`, 12) are the regression guard.

- [ ] **Step 1: Read the current file** so you preserve everything (constants `_TRANSFORMS`, `_X`, `_TOL`, `_SAFE_GLOBAL`, `_real_solutions`, `VerificationResult`).

- [ ] **Step 2: Add a public `parse_sides` and refactor `verify_equation` to use it.** Insert this function ABOVE `verify_equation`:
```python
def parse_sides(equation_str: str):
    """Parse 'lhs = rhs' into (lhs, rhs) sympy expressions using the sandboxed
    namespace. Raises ValueError if the string is not a single well-formed
    equation (wrong count of '=', inequality operators, or parse failure)."""
    s = (equation_str or "").strip()
    if s.count("=") != 1 or any(op in s for op in ("!=", "<=", ">=")):
        raise ValueError("equation must contain exactly one '='")
    lhs_str, rhs_str = s.split("=")
    try:
        lhs = parse_expr(lhs_str, transformations=_TRANSFORMS,
                         local_dict={"x": _X}, global_dict=_SAFE_GLOBAL, evaluate=True)
        rhs = parse_expr(rhs_str, transformations=_TRANSFORMS,
                         local_dict={"x": _X}, global_dict=_SAFE_GLOBAL, evaluate=True)
    except Exception as e:
        raise ValueError(f"parse error: {e}")
    return lhs, rhs
```
Then replace the body of `verify_equation` UP TO the solve step with:
```python
def verify_equation(equation_str: str, target: float) -> VerificationResult:
    try:
        lhs, rhs = parse_sides(equation_str)
    except ValueError as e:
        return VerificationResult(False, False, False, [], str(e))

    if _X not in (lhs - rhs).free_symbols:
        return VerificationResult(False, False, False, [],
                                  "equation does not constrain x (x absent or cancels out)")

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
(The `<=`/`!=`/`>=` guard and parse errors now live in `parse_sides`; the messages are unchanged, so existing tests still pass.)

- [ ] **Step 3: Run the existing verifier tests (regression)**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_math_equation_verifier.py -q` → expect `12 passed` (unchanged behavior).

- [ ] **Step 4: Commit**
```bash
git -C "D:/Inversa" add src/inversa/verifiers/math_equation.py
git -C "D:/Inversa" commit -m "refactor(verifier): extract public parse_sides (DRY, behavior-preserving)"
```

---

### Task 2: `difficulty()` (polynomial-degree band)

**Files:** Create `src/inversa/verifiers/difficulty.py`; Test `tests/test_difficulty.py`.

- [ ] **Step 1: Write the failing tests**
`tests/test_difficulty.py`:
```python
from inversa.verifiers.difficulty import difficulty


def test_linear_is_band_1():
    r = difficulty("2*x + 1 = 7")
    assert r.band == 1
    assert r.degree == 1
    assert r.non_polynomial is False


def test_quadratic_is_band_2():
    r = difficulty("x**2 - 4 = 0")
    assert r.band == 2 and r.degree == 2


def test_cubic_is_band_3():
    r = difficulty("x**3 - x = 0")
    assert r.band == 3 and r.degree == 3


def test_high_degree_clamped_to_5():
    r = difficulty("x**7 = 1")
    assert r.band == 5 and r.degree == 7


def test_non_polynomial_is_band_5():
    r = difficulty("cos(x) = x")
    assert r.band == 5 and r.non_polynomial is True


def test_identity_is_band_0():
    r = difficulty("x = x")
    assert r.band == 0 and r.error is not None


def test_unparseable_is_band_0():
    r = difficulty("x + = 3")
    assert r.band == 0 and r.error is not None
```

- [ ] **Step 2: Run to verify fail**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_difficulty.py -q` → FAIL (`No module named 'inversa.verifiers.difficulty'`).

- [ ] **Step 3: Implement**
`src/inversa/verifiers/difficulty.py`:
```python
"""Machine difficulty proxy for a posed equation: polynomial degree in x (v1).

band = polynomial degree of (lhs - rhs) in x, clamped to 1..5; non-polynomial
(transcendental) equations are treated as maximum difficulty (band 5). band 0
means "not measurable" (parse failure or the equation reduces to an identity).
This is a transparent v1 proxy — richer signals (operation count, solution
class, solver step-count) are future work (design §12). Glass-box: degree and
the non-polynomial flag are returned as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp

from inversa.verifiers.math_equation import parse_sides

_X = sp.Symbol("x")


@dataclass
class DifficultyResult:
    band: int                   # 1..5; 0 if not measurable
    degree: Optional[int]       # polynomial degree in x, or None if non-polynomial
    non_polynomial: bool
    error: Optional[str] = None


def difficulty(equation_str: str) -> DifficultyResult:
    try:
        lhs, rhs = parse_sides(equation_str)
    except ValueError as e:
        return DifficultyResult(0, None, False, str(e))

    expr = sp.expand(lhs - rhs)
    if expr == 0:
        return DifficultyResult(0, None, False, "identity (x cancels; no constraint)")

    try:
        deg = int(sp.degree(expr, gen=_X))
    except (sp.PolynomialError, TypeError, ValueError):
        return DifficultyResult(5, None, True)

    if deg <= 0:
        return DifficultyResult(0, deg, False, "no x term")
    return DifficultyResult(max(1, min(5, deg)), deg, False)
```

- [ ] **Step 4: Run to verify pass**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_difficulty.py -q` → expect `7 passed`.

- [ ] **Step 5: Commit**
```bash
git -C "D:/Inversa" add src/inversa/verifiers/difficulty.py tests/test_difficulty.py
git -C "D:/Inversa" commit -m "feat(difficulty): polynomial-degree difficulty band (v1)"
```

---

### Task 3: Calibration runner

**Files:** Create `src/inversa/tasks/calibration.py`; Test `tests/test_calibration.py`.

- [ ] **Step 1: Write the failing tests**
`tests/test_calibration.py`:
```python
from inversa.adapters.fake import FakeAdapter
from inversa.tasks.calibration import (
    build_calibration_prompt,
    run_calibration_item,
    run_calibration_batch,
)


def test_prompt_mentions_target_and_level():
    p = build_calibration_prompt(3, 2)
    assert "3" in p and "2" in p and "x" in p


def test_perfect_calibration_when_degree_matches():
    # requested degree 2, model returns a quadratic whose solution is 3
    adapter = FakeAdapter(["(x - 3)*(x - 3) = 0"])
    item = run_calibration_item(adapter, 3, 2)
    assert item.valid is True
    assert item.difficulty.band == 2
    assert item.calibration_error == 0


def test_miscalibrated_when_degree_wrong():
    # requested degree 3 but model returns a linear equation (degree 1), still solves to 3
    adapter = FakeAdapter(["2*x = 6"])
    item = run_calibration_item(adapter, 3, 3)
    assert item.valid is True
    assert item.difficulty.band == 1
    assert item.calibration_error == 2  # |3 - 1|


def test_invalid_equation_has_no_calibration_credit():
    # solves to 3, but target is 5 -> invalid; calibration_error still computed from band,
    # but report layer (Task 4) only counts valid items.
    adapter = FakeAdapter(["2*x = 6"])
    item = run_calibration_item(adapter, 5, 1)
    assert item.valid is False


def test_batch_size_is_targets_times_levels():
    adapter = FakeAdapter(["x = 1"])
    items = run_calibration_batch(adapter, [1, 2], [1, 2, 3])
    assert len(items) == 6
```

- [ ] **Step 2: Run to verify fail**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_calibration.py -q` → FAIL (`No module named 'inversa.tasks.calibration'`).

- [ ] **Step 3: Implement**
`src/inversa/tasks/calibration.py`:
```python
"""Difficulty-targeted posing: ask the model to pose at a requested degree and
measure validity + calibration (|requested - actual degree band|)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from inversa.adapters.base import Adapter
from inversa.tasks.posing import _extract_equation
from inversa.verifiers.difficulty import difficulty, DifficultyResult
from inversa.verifiers.math_equation import verify_equation, VerificationResult

CALIB_PROMPT = (
    "You are constructing a math problem.\n"
    "Write a single-variable algebraic EQUATION in the variable x whose unique "
    "solution is exactly {target} AND whose polynomial degree in x is exactly {level}.\n"
    "Output ONLY the equation as '<lhs> = <rhs>', with no explanation. "
    "Use sympy syntax (e.g. x**2 - 4 = 0 has degree 2)."
)


def build_calibration_prompt(target, level: int) -> str:
    return CALIB_PROMPT.format(target=target, level=level)


@dataclass
class CalibrationItem:
    target: float
    requested_level: int
    raw_output: str
    equation: str
    verification: VerificationResult
    difficulty: DifficultyResult
    valid: bool                       # solves to target
    calibration_error: Optional[int]  # |requested_level - actual band|, None if band==0


def run_calibration_item(adapter: Adapter, target, level: int) -> CalibrationItem:
    raw = adapter.generate(build_calibration_prompt(target, level))
    equation = _extract_equation(raw)
    ver = verify_equation(equation, target)
    dif = difficulty(equation)
    cal_err = abs(int(level) - dif.band) if dif.band else None
    return CalibrationItem(
        target=float(target),
        requested_level=int(level),
        raw_output=raw,
        equation=equation,
        verification=ver,
        difficulty=dif,
        valid=ver.valid,
        calibration_error=cal_err,
    )


def run_calibration_batch(adapter: Adapter, targets, levels) -> list:
    return [run_calibration_item(adapter, t, lv) for t in targets for lv in levels]
```

- [ ] **Step 4: Run to verify pass**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_calibration.py -q` → expect `5 passed`.

- [ ] **Step 5: Commit**
```bash
git -C "D:/Inversa" add src/inversa/tasks/calibration.py tests/test_calibration.py
git -C "D:/Inversa" commit -m "feat(calibration): difficulty-targeted posing runner"
```

---

### Task 4: Calibration report

**Files:** Modify `src/inversa/report.py`; Test `tests/test_report_calibration.py`.

- [ ] **Step 1: Write the failing tests**
`tests/test_report_calibration.py`:
```python
from inversa.adapters.fake import FakeAdapter
from inversa.tasks.calibration import run_calibration_batch
from inversa.report import calibration_summary, format_calibration_summary


def test_summary_counts_and_mace():
    # target 3, levels [1, 3], model always returns "2*x = 6" (valid, band 1)
    # level 1 -> err 0 ; level 3 -> err 2 ; mean = 1.0
    adapter = FakeAdapter(["2*x = 6", "2*x = 6"])
    items = run_calibration_batch(adapter, [3], [1, 3])
    s = calibration_summary(items)
    assert s.n == 2
    assert s.n_valid == 2
    assert s.n_measurable == 2
    assert s.mean_abs_calibration_error == 1.0


def test_summary_empty():
    s = calibration_summary([])
    assert s.n == 0
    assert s.mean_abs_calibration_error is None


def test_format_is_readable():
    adapter = FakeAdapter(["2*x = 6", "2*x = 6"])
    items = run_calibration_batch(adapter, [3], [1, 3])
    text = format_calibration_summary(calibration_summary(items))
    assert "calibration" in text.lower()
```

- [ ] **Step 2: Run to verify fail**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_report_calibration.py -q` → FAIL (`cannot import name 'calibration_summary'`).

- [ ] **Step 3: Implement — APPEND to `src/inversa/report.py`** (keep existing `Summary`/`summarize`/`format_summary`):
```python
@dataclass
class CalibrationSummary:
    n: int
    n_valid: int
    validity_rate: float
    n_measurable: int                            # valid AND difficulty band measurable
    mean_abs_calibration_error: float | None     # over measurable items, else None


def calibration_summary(items) -> CalibrationSummary:
    n = len(items)
    n_valid = sum(1 for it in items if it.valid)
    measurable = [it for it in items
                  if it.valid and it.calibration_error is not None]
    mace = (sum(it.calibration_error for it in measurable) / len(measurable)
            if measurable else None)
    return CalibrationSummary(
        n=n,
        n_valid=n_valid,
        validity_rate=(n_valid / n if n else 0.0),
        n_measurable=len(measurable),
        mean_abs_calibration_error=mace,
    )


def format_calibration_summary(s: CalibrationSummary) -> str:
    mace = "n/a" if s.mean_abs_calibration_error is None else f"{s.mean_abs_calibration_error:.2f}"
    return (f"items={s.n} valid={s.n_valid} ({s.validity_rate:.1%}) "
            f"measurable={s.n_measurable} mean_abs_calibration_error={mace}")
```

- [ ] **Step 4: Run to verify pass + full suite**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_report_calibration.py -q` → expect `3 passed`.
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests -q` → expect all green.

- [ ] **Step 5: Commit**
```bash
git -C "D:/Inversa" add src/inversa/report.py tests/test_report_calibration.py
git -C "D:/Inversa" commit -m "feat(report): calibration summary (validity + mean abs calibration error)"
```

---

### Task 5: Calibration CLI + live smoke

**Files:** Create `src/inversa/cli_calibration.py`; Test `tests/test_cli_calibration.py`. Reuse bank `data/banks/math_targets_slice1.json`.

- [ ] **Step 1: Write the failing test** (bank/level validation path — runs before any adapter/network)
`tests/test_cli_calibration.py`:
```python
import json

import pytest

from inversa.cli_calibration import main


def test_rejects_non_integer_levels(tmp_path):
    bank = tmp_path / "b.json"
    bank.write_text(json.dumps({"targets": [3]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--bank", str(bank), "--levels", "abc"])


def test_rejects_bank_without_targets(tmp_path):
    bank = tmp_path / "b.json"
    bank.write_text(json.dumps({"domain": "math_equation"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--bank", str(bank), "--levels", "1,2"])
```

- [ ] **Step 2: Run to verify fail**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_cli_calibration.py -q` → FAIL (`No module named 'inversa.cli_calibration'`).

- [ ] **Step 3: Implement**
`src/inversa/cli_calibration.py`:
```python
"""CLI: run a difficulty-calibration batch from a target bank and print a report."""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from inversa.adapters.anthropic_adapter import AnthropicAdapter
from inversa.report import calibration_summary, format_calibration_summary
from inversa.tasks.calibration import run_calibration_batch


def main(argv=None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Inversa difficulty-calibration runner")
    parser.add_argument("--bank", required=True, help="path to target bank JSON")
    parser.add_argument("--levels", default="1,2,3,4",
                        help="comma-separated requested polynomial degrees")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--verbose", action="store_true",
                        help="also print each item's raw model output")
    args = parser.parse_args(argv)

    with open(args.bank, encoding="utf-8") as f:
        data = json.load(f)
    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        parser.error(f"bank {args.bank!r} must contain a non-empty 'targets' list")
    try:
        levels = [int(x) for x in args.levels.split(",")]
    except ValueError:
        parser.error("--levels must be comma-separated integers, e.g. 1,2,3,4")

    adapter = AnthropicAdapter(model=args.model)
    items = run_calibration_batch(adapter, targets, levels)

    # Glass-box output (design §3.5): per-item requested vs actual degree + verdict.
    print(format_calibration_summary(calibration_summary(items)))
    for it in items:
        d = it.difficulty
        print(f"  target={it.target} req_deg={it.requested_level} eq={it.equation!r} "
              f"actual_deg={d.degree} band={d.band} valid={it.valid} "
              f"cal_err={it.calibration_error}")
        if args.verbose:
            print(f"    raw_output={it.raw_output!r}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify**
Run: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests/test_cli_calibration.py -q` → expect `2 passed`.
Run: `D:/Inversa/.venv/Scripts/python.exe -c "import inversa.cli_calibration; print('ok')"` → `ok`.
Run the FULL suite: `D:/Inversa/.venv/Scripts/python.exe -m pytest D:/Inversa/tests -q` → expect all green.

- [ ] **Step 5: Commit**
```bash
git -C "D:/Inversa" add src/inversa/cli_calibration.py tests/test_cli_calibration.py
git -C "D:/Inversa" commit -m "feat(cli): difficulty-calibration runner CLI"
```

- [ ] **Step 6: Live smoke (user-run, needs ANTHROPIC_API_KEY in .env)**
```
D:\Inversa\.venv\Scripts\python.exe -m inversa.cli_calibration --bank D:\Inversa\data\banks\math_targets_slice1.json --levels 1,2,3,4 --verbose
```
Expected: a summary line `items=... valid=... (..%) measurable=... mean_abs_calibration_error=...` then per-item `req_deg` vs `actual_deg`/`band`, `valid`, `cal_err`. **Interpretation:** low mean_abs_calibration_error + high validity ⇒ the model can deliberately control equation degree while keeping the answer — a difficulty-understanding signal. Watch for: validity dropping at higher requested degrees, or systematic under/over-shooting of degree. These feed slice 2b (adversarial) and slice 3 (θ_gen vs θ_solve).

---

## Self-Review

**Spec coverage:** difficulty proxy (Task 2) ✓; difficulty-target posing + calibration error (Task 3) ✓; calibration report/MACE (Task 4) ✓; CLI + glass-box output + live smoke (Task 5) ✓; DRY parse reuse (Task 1) ✓. Adversarial/IRT correctly deferred. ✓

**Placeholder scan:** none — every step has complete code + exact commands/expected output. ✓

**Type consistency:** `DifficultyResult(band, degree, non_polynomial, error)` used identically in difficulty.py, calibration.py (`it.difficulty.band`/`.degree`), cli (`d.degree`/`d.band`). `CalibrationItem(target, requested_level, raw_output, equation, verification, difficulty, valid, calibration_error)` used in report (`it.valid`, `it.calibration_error`) and cli. `parse_sides` raises `ValueError`; both `verify_equation` and `difficulty` catch `ValueError`. `calibration_summary` fields match `format_calibration_summary` + tests. ✓
