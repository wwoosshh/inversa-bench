from inversa.adapters.fake import FakeAdapter
from inversa.tasks.adversarial import (
    build_poser_prompt,
    build_solver_prompt,
    parse_numeric_answer,
    run_adversarial_item,
    run_adversarial_batch,
)


def test_poser_prompt_mentions_target_and_x():
    p = build_poser_prompt(3)
    assert "3" in p and "x" in p


def test_solver_prompt_contains_equation():
    assert "2*x = 6" in build_solver_prompt("2*x = 6")


def test_parse_plain_integer():
    assert parse_numeric_answer("3") == 3.0


def test_parse_x_equals_form():
    assert parse_numeric_answer("x = 7") == 7.0
    assert parse_numeric_answer("x=-5") == -5.0


def test_parse_fraction():
    assert parse_numeric_answer("1/2") == 0.5


def test_parse_embedded_in_prose():
    assert parse_numeric_answer("The solution is 42") == 42.0


def test_parse_no_number_returns_none():
    assert parse_numeric_answer("no real solution") is None


def test_adversarial_success_when_weak_wrong():
    poser = FakeAdapter(["(x - 3)*(x**2 + 1) = 0"])  # valid for target 3, only real root is 3
    weak = FakeAdapter(["5"])                         # weak model answers wrong
    item = run_adversarial_item(poser, weak, 3)
    assert item.valid is True
    assert item.weak_answer == 5.0
    assert item.weak_correct is False
    assert item.adversarial_success is True


def test_no_adversarial_when_weak_correct():
    poser = FakeAdapter(["(x - 3)*(x**2 + 1) = 0"])
    weak = FakeAdapter(["x = 3"])
    item = run_adversarial_item(poser, weak, 3)
    assert item.valid is True
    assert item.weak_correct is True
    assert item.adversarial_success is False


def test_invalid_pose_is_never_adversarial_success():
    poser = FakeAdapter(["2*x = 6"])  # solves to 3, not 5 -> invalid for target 5
    weak = FakeAdapter(["999"])
    item = run_adversarial_item(poser, weak, 5)
    assert item.valid is False
    assert item.adversarial_success is False


def test_batch_size():
    poser = FakeAdapter(["x = 1"])
    weak = FakeAdapter(["0"])
    items = run_adversarial_batch(poser, weak, [1, 2, 3])
    assert len(items) == 3


def test_parse_x_equals_wins_over_trailing_number():
    assert parse_numeric_answer("We get x = 5 after step 2") == 5.0


def test_parse_x_equals_ignores_check_annotation():
    assert parse_numeric_answer("x = 7 (check: 2*7 = 14)") == 7.0


def test_parse_no_solution_phrase_variants():
    assert parse_numeric_answer("There is no real solution") is None
    assert parse_numeric_answer("no solution") is None
