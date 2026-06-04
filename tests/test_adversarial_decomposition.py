"""#2: adversarial posing decomposed into two axes — pose VALIDITY (target is the
UNIQUE real solution, the actual constraint) and DIFFICULTY (fool-rate among valid
poses). A valid-but-non-unique equation no longer counts as adversarial success."""
from inversa.adapters.fake import FakeAdapter
from inversa.experiment import ModelScores, correlations, evaluate_model
from inversa.report import adversarial_summary
from inversa.tasks.adversarial import run_adversarial_item, run_adversarial_batch


def test_non_unique_valid_pose_is_not_adversarial_success():
    # roots are 3 AND -1 -> target 3 is a member but NOT unique; weak gets it wrong.
    poser = FakeAdapter(["(x - 3)*(x + 1) = 0"])
    weak = FakeAdapter(["9"])
    item = run_adversarial_item(poser, weak, 3)
    assert item.valid is True        # 3 is a real solution (membership)
    assert item.unique is False      # but not the unique one
    assert item.weak_correct is False
    assert item.adversarial_success is False  # uniqueness constraint not met -> no credit


def test_unique_valid_and_weak_wrong_is_success():
    poser = FakeAdapter(["(x - 3)*(x**2 + 1) = 0"])  # only real root is 3
    weak = FakeAdapter(["9"])
    item = run_adversarial_item(poser, weak, 3)
    assert item.unique is True
    assert item.adversarial_success is True


def test_summary_decomposes_validity_and_fool_rate():
    # item0: unique + weak wrong (success); item1: non-unique + weak wrong (not success);
    # item2: unique + weak correct (valid but not hard).
    poser = FakeAdapter(["(x - 3)*(x**2 + 1) = 0", "(x - 3)*(x + 1) = 0", "(x - 3)*(x**2 + 1) = 0"])
    weak = FakeAdapter(["9", "9", "3"])
    items = run_adversarial_batch(poser, weak, [3, 3, 3])
    s = adversarial_summary(items)
    assert s.n == 3
    assert s.n_unique == 2
    assert s.pose_validity_rate == 2 / 3
    assert s.n_adversarial_success == 1
    assert s.adversarial_success_rate == 1 / 3
    assert s.fool_rate_on_unique == 0.5   # 1 of 2 unique poses fooled the weak solver


def test_fool_rate_none_when_no_unique_pose():
    poser = FakeAdapter(["(x - 3)*(x + 1) = 0"])  # never unique for target 3
    weak = FakeAdapter(["9"])
    s = adversarial_summary(run_adversarial_batch(poser, weak, [3]))
    assert s.pose_validity_rate == 0.0
    assert s.fool_rate_on_unique is None


def test_correlations_expose_decomposed_axes():
    scores = [
        ModelScores("a", 0.9, 0.1, 0.2, pose_validity_rate=0.2, pose_fool_rate=1.0),
        ModelScores("b", 0.6, 0.5, 0.5, pose_validity_rate=0.5, pose_fool_rate=1.0),
        ModelScores("c", 0.3, 0.9, 0.8, pose_validity_rate=0.8, pose_fool_rate=1.0),
    ]
    cors = correlations(scores)
    assert "solve_vs_pose_validity" in cors
    assert "solve_vs_pose_fool_rate" in cors
    assert cors["solve_vs_pose_validity"] == -1.0  # validity rises as solve falls


def test_evaluate_model_populates_decomposed_scores():
    poser = FakeAdapter(["3", "2*x = 6", "(x - 3)*(x**2 + 1) = 0"])
    weak = FakeAdapter(["999"])
    scores = evaluate_model(
        "fake", poser, weak,
        solve_problems=[{"equation": "2*x + 1 = 7", "answer": 3}],
        calib_targets=[3], calib_levels=[1], adv_targets=[3],
    )
    assert 0.0 <= scores.pose_validity_rate <= 1.0
    assert scores.pose_fool_rate is None or 0.0 <= scores.pose_fool_rate <= 1.0
