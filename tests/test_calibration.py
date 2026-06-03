from inversa.adapters.fake import FakeAdapter
from inversa.tasks.calibration import (
    build_calibration_prompt,
    run_calibration_item,
    run_calibration_batch,
)


def test_prompt_mentions_target_and_level():
    p = build_calibration_prompt(3, 2)
    assert "3" in p and "2" in p and "x" in p


def test_perfect_calibration_when_degree_matches():
    adapter = FakeAdapter(["(x - 3)*(x - 3) = 0"])  # solution 3, degree 2
    item = run_calibration_item(adapter, 3, 2)
    assert item.valid is True
    assert item.difficulty.band == 2
    assert item.calibration_error == 0


def test_miscalibrated_when_degree_wrong():
    adapter = FakeAdapter(["2*x = 6"])  # solution 3, degree 1
    item = run_calibration_item(adapter, 3, 3)
    assert item.valid is True
    assert item.difficulty.band == 1
    assert item.calibration_error == 2  # |3 - 1|


def test_invalid_equation():
    adapter = FakeAdapter(["2*x = 6"])  # solves to 3, not 5
    item = run_calibration_item(adapter, 5, 1)
    assert item.valid is False


def test_batch_size_is_targets_times_levels():
    adapter = FakeAdapter(["x = 1"])
    items = run_calibration_batch(adapter, [1, 2], [1, 2, 3])
    assert len(items) == 6
