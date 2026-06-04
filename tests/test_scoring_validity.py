"""Scoring-validity fixes: calibration uses raw polynomial degree (not the capped
1..5 band), and the solve-answer parser prefers an explicit '#### <num>' final line
so verbose / reasoning models are scored on their actual answer."""
from inversa.adapters.fake import FakeAdapter
from inversa.tasks.adversarial import parse_numeric_answer
from inversa.tasks.calibration import run_calibration_item


def test_calibration_uses_raw_degree_not_capped_band():
    # degree-6 equation requested at level 6 -> cal_err 0 (raw degree), NOT 1 (band saturates at 5)
    item = run_calibration_item(FakeAdapter(["x**6 = 0"]), 0, 6)
    assert item.valid is True
    assert item.difficulty.degree == 6
    assert item.calibration_error == 0


def test_calibration_level_above_band_cap_measures_real_gap():
    # degree-2 equation requested at level 6 -> cal_err |6-2| = 4
    item = run_calibration_item(FakeAdapter(["x**2 - 4 = 0"]), 2, 6)
    assert item.difficulty.degree == 2
    assert item.calibration_error == 4


def test_parse_final_marker_wins_over_reasoning_numbers():
    assert parse_numeric_answer("Step 1: x=0 is extraneous, so the answer is\n#### 5") == 5.0
    assert parse_numeric_answer("lots of working 2*3=6 ...\n#### -2") == -2.0


def test_parse_without_marker_still_works():
    assert parse_numeric_answer("3") == 3.0
    assert parse_numeric_answer("x = 7") == 7.0
    assert parse_numeric_answer("no real solution") is None
