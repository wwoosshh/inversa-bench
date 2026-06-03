from inversa.adapters.fake import FakeAdapter
from inversa.experiment import ModelScores, evaluate_model, correlations


def test_evaluate_model_assembles_scores():
    poser = FakeAdapter(["3", "2*x = 6", "2*x = 6"])
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
    assert cors["solve_vs_calibration_mace"] == -1.0
    assert cors["solve_vs_adversarial"] == -1.0


def test_evaluate_model_detailed_keeps_items():
    from inversa.experiment import evaluate_model_detailed, ModelRun
    poser = FakeAdapter(["3", "2*x = 6", "2*x = 6"])
    weak = FakeAdapter(["999"])
    run = evaluate_model_detailed(
        "fake", poser, weak,
        solve_problems=[{"equation": "2*x + 1 = 7", "answer": 3}],
        calib_targets=[3], calib_levels=[1], adv_targets=[3],
    )
    assert isinstance(run, ModelRun)
    assert run.model == "fake"
    assert len(run.solve_items) == 1
    assert len(run.calibration_items) == 1
    assert len(run.adversarial_items) == 1
    assert run.scores.solve_accuracy == 1.0
