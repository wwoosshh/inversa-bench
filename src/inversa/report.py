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
