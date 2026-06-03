from inversa.adapters.fake import FakeAdapter
from inversa.tasks.posing import run_batch
from inversa.report import summarize, format_summary


def _results():
    # two valid (x=1, x=2), one invalid for target 9 (adapter returns x=1)
    adapter = FakeAdapter(["x = 1", "x = 2", "x = 1"])
    return run_batch(adapter, [1, 2, 9])


def test_summarize_counts():
    s = summarize(_results())
    assert s.n == 3
    assert s.n_valid == 2
    assert s.validity_rate == 2 / 3


def test_summarize_empty():
    s = summarize([])
    assert s.n == 0
    assert s.validity_rate == 0.0


def test_format_summary_is_readable():
    s = summarize(_results())
    text = format_summary(s)
    assert "valid" in text
    assert "3" in text
