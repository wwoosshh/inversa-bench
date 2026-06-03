from inversa.adapters.fake import FakeAdapter
from inversa.tasks.solving import solve_item, solve_batch, accuracy


def test_solve_item_correct():
    a = FakeAdapter(["3"])
    it = solve_item(a, "2*x + 1 = 7", 3)
    assert it.answer == 3.0 and it.correct is True


def test_solve_item_wrong():
    a = FakeAdapter(["99"])
    it = solve_item(a, "2*x + 1 = 7", 3)
    assert it.correct is False


def test_solve_item_x_equals_reply():
    a = FakeAdapter(["x = 7"])
    it = solve_item(a, "3*x - 5 = 16", 7)
    assert it.correct is True


def test_accuracy():
    a = FakeAdapter(["3", "99", "7"])
    probs = [
        {"equation": "2*x + 1 = 7", "answer": 3},
        {"equation": "2*x + 1 = 7", "answer": 3},
        {"equation": "3*x - 5 = 16", "answer": 7},
    ]
    items = solve_batch(a, probs)
    assert accuracy(items) == 2 / 3
