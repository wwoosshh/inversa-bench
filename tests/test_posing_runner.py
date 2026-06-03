from inversa.adapters.fake import FakeAdapter
from inversa.tasks.posing import build_prompt, _extract_equation, run_item, run_batch


def test_build_prompt_mentions_target_and_x():
    p = build_prompt(3)
    assert "3" in p
    assert "x" in p


def test_extract_equation_takes_first_line_with_equals():
    raw = "Here you go:\n2*x + 1 = 7\nthanks"
    assert _extract_equation(raw) == "2*x + 1 = 7"


def test_extract_equation_strips_backticks():
    assert _extract_equation("`x = 5`") == "x = 5"


def test_run_item_valid_equation():
    adapter = FakeAdapter(["2*x = 6"])
    result = run_item(adapter, 3)
    assert result.target == 3.0
    assert result.equation == "2*x = 6"
    assert result.verification.valid is True


def test_run_item_invalid_equation():
    adapter = FakeAdapter(["2*x = 6"])
    result = run_item(adapter, 5)  # 2x=6 solves to 3, not 5
    assert result.verification.valid is False


def test_run_batch_returns_one_result_per_target():
    adapter = FakeAdapter(["x = 1", "x = 2", "x = 3"])
    results = run_batch(adapter, [1, 2, 3])
    assert len(results) == 3
    assert all(r.verification.valid for r in results)
