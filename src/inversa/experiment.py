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
    # Decomposed generative axes (defaults keep older positional construction working):
    pose_validity_rate: float = 0.0           # fraction of posed eqs with UNIQUE solution = target
    pose_fool_rate: Optional[float] = None    # among unique-valid, fraction the weak solver got WRONG


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
    adv_sum = adversarial_summary(adv_items)
    scores = ModelScores(
        model=model,
        solve_accuracy=accuracy(solve_items),
        calibration_mace=calibration_summary(calib_items).mean_abs_calibration_error,
        adversarial_success_rate=adv_sum.adversarial_success_rate,
        pose_validity_rate=adv_sum.pose_validity_rate,
        pose_fool_rate=adv_sum.fool_rate_on_unique,
    )
    return ModelRun(model, scores, solve_items, calib_items, adv_items)


def correlations(scores: List[ModelScores]) -> dict:
    solve = [s.solve_accuracy for s in scores]
    adv = [s.adversarial_success_rate for s in scores]
    pose_validity = [s.pose_validity_rate for s in scores]
    mace_pairs = [(s.solve_accuracy, s.calibration_mace) for s in scores
                  if s.calibration_mace is not None]
    mace_corr = (spearman([p[0] for p in mace_pairs], [p[1] for p in mace_pairs])
                 if len(mace_pairs) >= 2 else None)
    # Among models that posed >=1 valid problem, does difficulty (fool-rate) track solving?
    fool_pairs = [(s.solve_accuracy, s.pose_fool_rate) for s in scores
                  if s.pose_fool_rate is not None]
    fool_corr = (spearman([p[0] for p in fool_pairs], [p[1] for p in fool_pairs])
                 if len(fool_pairs) >= 2 else None)
    return {
        "solve_vs_calibration_mace": mace_corr,
        "solve_vs_adversarial": spearman(solve, adv),
        "solve_vs_pose_validity": spearman(solve, pose_validity),
        "solve_vs_pose_fool_rate": fool_corr,
    }
