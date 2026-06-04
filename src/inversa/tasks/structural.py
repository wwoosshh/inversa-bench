"""Level-3 generative probes: does the model *construct/transform* mathematical
structure (understanding), or merely retrieve memorized problem templates (coverage)?

Recall is defeated by CONSTRUCTION: every task pins the required output to information
supplied at test time (an irrational/structural target, or a freshly-generated source
equation) so no memorized problem can satisfy it. All scoring is sympy forward-verified.

Form A (struct-pose): pose an equation whose UNIQUE real solution is a given irrational/
structural target. Templating breaks here — e.g. the minimal polynomial of sqrt(2)+1 also
carries the conjugate root 1-sqrt(2), violating uniqueness — so success needs real construction.

Form B (transform): given a random source equation E with unique real root r, construct a
NEW equation whose unique real root is g(r) (r+1, 2r, r**2, ...). Because E is random and r
is ugly, the answer must come from STRUCTURAL manipulation (e.g. substitute x -> x-1), not a
numeric lookup — the purest recall-proof test of "can it induce a structural change?".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from inversa.adapters.base import Adapter
from inversa.tasks.posing import extract_equation
from inversa.verifiers.math_equation import verify_equation, VerificationResult

STRUCT_POSE_PROMPT = (
    "Construct a single-variable equation in x whose UNIQUE real solution is exactly {target_desc}.\n"
    "There must be no other real solution. Radicals, fractions, exponentials and logs are allowed "
    "and encouraged.\n"
    "Output ONLY the equation as '<lhs> = <rhs>' in sympy syntax, with no explanation."
)

TRANSFORM_PROMPT = (
    "Let r be the UNIQUE real solution of this equation:\n{source}\n"
    "Do NOT solve for r numerically. By transforming the structure, construct a NEW single-variable "
    "equation in x whose unique real solution is exactly {g_desc} (where r is that solution above).\n"
    "Output ONLY the new equation as '<lhs> = <rhs>' in sympy syntax, with no explanation."
)


def build_struct_pose_prompt(target_desc: str) -> str:
    return STRUCT_POSE_PROMPT.format(target_desc=target_desc)


def build_transform_prompt(source: str, g_desc: str) -> str:
    return TRANSFORM_PROMPT.format(source=source, g_desc=g_desc)


@dataclass
class StructPoseItem:
    target_desc: str          # how the target was described to the model (e.g. "sqrt(2) + 1")
    target_value: float       # its numeric value, used for forward-verification
    novelty: str              # ladder rung label (e.g. "integer", "quadratic-irrational")
    raw_output: str
    equation: str
    verification: VerificationResult
    valid: bool               # target is the UNIQUE real solution (level-3 success)


def run_struct_pose_item(adapter: Adapter, target_desc: str, target_value,
                         novelty: str = "") -> StructPoseItem:
    raw = adapter.generate(build_struct_pose_prompt(target_desc))
    equation = extract_equation(raw)
    ver = verify_equation(equation, float(target_value))
    return StructPoseItem(target_desc, float(target_value), novelty, raw, equation, ver, ver.unique)


@dataclass
class TransformItem:
    source: str               # the random source equation E
    g_desc: str               # transform requested, e.g. "r + 1"
    r_value: float            # unique real root of E
    target_value: float       # g(r), what E' must have as its unique real root
    raw_output: str
    equation: str
    verification: VerificationResult
    valid: bool               # E' has target_value as its UNIQUE real solution


def run_transform_item(adapter: Adapter, source: str, g_desc: str,
                       r_value, target_value) -> TransformItem:
    raw = adapter.generate(build_transform_prompt(source, g_desc))
    equation = extract_equation(raw)
    ver = verify_equation(equation, float(target_value))
    return TransformItem(source, g_desc, float(r_value), float(target_value),
                         raw, equation, ver, ver.unique)


def validity_rate(items) -> float:
    """Fraction of items whose UNIQUE-solution constraint was met. Both item types expose .valid."""
    n = len(items)
    return (sum(1 for it in items if it.valid) / n) if n else 0.0
