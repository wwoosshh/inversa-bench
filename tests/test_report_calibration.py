from inversa.adapters.fake import FakeAdapter
from inversa.tasks.calibration import run_calibration_batch
from inversa.report import calibration_summary, format_calibration_summary


def test_summary_counts_and_mace():
    # target 3, levels [1, 3], model always returns "2*x = 6" (valid, band 1)
    # level 1 -> err 0 ; level 3 -> err 2 ; mean = 1.0
    adapter = FakeAdapter(["2*x = 6", "2*x = 6"])
    items = run_calibration_batch(adapter, [3], [1, 3])
    s = calibration_summary(items)
    assert s.n == 2
    assert s.n_valid == 2
    assert s.n_measurable == 2
    assert s.mean_abs_calibration_error == 1.0


def test_summary_empty():
    s = calibration_summary([])
    assert s.n == 0
    assert s.mean_abs_calibration_error is None


def test_format_is_readable():
    adapter = FakeAdapter(["2*x = 6", "2*x = 6"])
    items = run_calibration_batch(adapter, [3], [1, 3])
    text = format_calibration_summary(calibration_summary(items))
    assert "calibration" in text.lower()


def test_summary_excludes_invalid_from_mace():
    # "2*x = 6" is valid for target 3 (band 1), invalid for target 5; only the valid one counts
    adapter = FakeAdapter(["2*x = 6", "2*x = 6"])
    items = run_calibration_batch(adapter, [3, 5], [1])  # 2 items, 1 valid
    s = calibration_summary(items)
    assert s.n == 2
    assert s.n_valid == 1
    assert s.n_measurable == 1
    assert s.mean_abs_calibration_error == 0.0
