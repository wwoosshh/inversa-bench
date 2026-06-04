"""Inverse/posing task: target answer -> model constructs an equation -> verify."""
from __future__ import annotations

import re
from dataclasses import dataclass

from inversa.adapters.base import Adapter
from inversa.verifiers.math_equation import parse_sides, verify_equation, VerificationResult

_MARKER_RE = re.compile(r"####\s*(.+)")
_LABEL_RE = re.compile(r"^\s*(the\s+)?(equation|answer|result|f\s*\(\s*x\s*\))\s*[:=]?\s*", re.I)
_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
        "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
_SUP_RE = re.compile("[" + "".join(_SUP) + "]+")


def _norm_unicode(s: str) -> str:
    """Normalize the unicode math glyphs models emit so sympy can parse them: superscript
    powers (x² -> x**2), and the unicode minus/times/dot/divide operators."""
    s = _SUP_RE.sub(lambda m: "**" + "".join(_SUP[c] for c in m.group(0)), s)
    return (s.replace("−", "-").replace("×", "*").replace("⋅", "*")
            .replace("÷", "/").replace("√", "sqrt"))


def _clean(s: str) -> str:
    """Strip the cosmetic noise weak/verbose models wrap equations in: LaTeX delimiters,
    code fences, \\cdot, a leading 'f(x) =' label, '^' caret powers (sympy reads '^' as XOR),
    and unicode math glyphs."""
    s = _norm_unicode(s.strip())
    for tok in ("$$", "$", "\\(", "\\)", "\\[", "\\]", "`"):
        s = s.replace(tok, "")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\,", "").replace("\\!", "")
    s = _LABEL_RE.sub("", s)
    s = re.sub(r"^[^=]*:\s*", "", s)  # strip a leading 'Final:' / 'So:' style prefix
    s = s.replace("^", "**")
    return s.strip().rstrip(".").strip()


def parses(s: str) -> bool:
    """True if `s` is a well-formed equation our verifier can parse (used to decide retries)."""
    try:
        parse_sides(s)
        return True
    except Exception:
        return False


_parses = parses  # backward-compatible alias

PROMPT_TEMPLATE = (
    "You are constructing a math problem.\n"
    "Write a single-variable algebraic EQUATION in the variable x whose unique "
    "solution is exactly {target}.\n"
    "You may reason first, but you MUST end with the equation on its own final line prefixed "
    "by '#### ', in Python/sympy syntax. Example: '#### x**2 - 9 = 0'.\n"
)


def build_prompt(target) -> str:
    return PROMPT_TEMPLATE.format(target=target)


def extract_equation(raw: str) -> str:
    """Pull the final equation from a model reply, robust to verbose/LaTeX output.
    Priority: '#### <eq>' marker (last wins) -> the last line that actually PARSES as an
    equation -> the last line containing '=' -> the whole reply. Each candidate is cleaned
    of LaTeX/caret noise first, so format quirks don't turn a correct answer into a false zero."""
    s = (raw or "").strip()
    cands = [m for m in reversed(_MARKER_RE.findall(s))]
    cands += [ln for ln in reversed(s.splitlines()) if "=" in ln]
    cands.append(s)
    cleaned = [_clean(c) for c in cands]
    # best: a candidate that parses as a real equation
    for c in cleaned:
        if "=" in c and _parses(c):
            return c
    # else: first candidate that at least contains '='
    for c in cleaned:
        if "=" in c:
            return c
    return _clean(s)


@dataclass
class ItemResult:
    target: float
    raw_output: str
    equation: str
    verification: VerificationResult


def run_item(adapter: Adapter, target) -> ItemResult:
    raw = adapter.generate(build_prompt(target))
    equation = extract_equation(raw)
    verification = verify_equation(equation, target)
    return ItemResult(float(target), raw, equation, verification)


def run_batch(adapter: Adapter, targets) -> list:
    return [run_item(adapter, t) for t in targets]
