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
