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
    return evaluate_model_detailed(model, poser, weak, solve_problems,
                                   calib_targets, calib_levels, adv_targets).scores


@dataclass
class ModelRun:
    model: str
    scores: ModelScores
    solve_items: list
    calibration_items: list
    adversarial_items: list


def evaluate_model_detailed(model: str, poser: Adapter, weak: Adapter,
                            solve_problems, calib_targets, calib_levels, adv_targets) -> ModelRun:
    solve_items = solve_batch(poser, solve_problems)
    calib_items = run_calibration_batch(poser, calib_targets, calib_levels)
    adv_items = run_adversarial_batch(poser, weak, adv_targets)
    scores = ModelScores(
        model=model,
        solve_accuracy=accuracy(solve_items),
        calibration_mace=calibration_summary(calib_items).mean_abs_calibration_error,
        adversarial_success_rate=adversarial_summary(adv_items).adversarial_success_rate,
    )
    return ModelRun(model, scores, solve_items, calib_items, adv_items)


def correlations(scores: List[ModelScores]) -> dict:
    solve = [s.solve_accuracy for s in scores]
    adv = [s.adversarial_success_rate for s in scores]
    mace_pairs = [(s.solve_accuracy, s.calibration_mace) for s in scores
                  if s.calibration_mace is not None]
    mace_corr = (spearman([p[0] for p in mace_pairs], [p[1] for p in mace_pairs])
                 if len(mace_pairs) >= 2 else None)
    return {
        "solve_vs_calibration_mace": mace_corr,
        "solve_vs_adversarial": spearman(solve, adv),
    }
