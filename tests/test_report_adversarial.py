from inversa.adapters.fake import FakeAdapter
from inversa.tasks.adversarial import run_adversarial_batch
from inversa.report import adversarial_summary, format_adversarial_summary


def test_summary_counts():
    # both poses valid (target 3); weak wrong then correct -> 1 adversarial success of 2
    poser = FakeAdapter(["(x - 3)*(x**2 + 1) = 0"])
    weak = FakeAdapter(["5", "3"])
    items = run_adversarial_batch(poser, weak, [3, 3])
    s = adversarial_summary(items)
    assert s.n == 2
    assert s.n_valid == 2
    assert s.n_adversarial_success == 1
    assert s.adversarial_success_rate == 0.5
    assert s.weak_solver_accuracy_on_valid == 0.5


def test_summary_empty():
    s = adversarial_summary([])
    assert s.n == 0
    assert s.adversarial_success_rate == 0.0
    assert s.weak_solver_accuracy_on_valid is None


def test_format_is_readable():
    poser = FakeAdapter(["(x - 3)*(x**2 + 1) = 0"])
    weak = FakeAdapter(["5", "3"])
    items = run_adversarial_batch(poser, weak, [3, 3])
    text = format_adversarial_summary(adversarial_summary(items))
    assert "adversarial" in text.lower()
