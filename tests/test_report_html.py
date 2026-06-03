from inversa.experiment import ModelScores
from inversa.report_html import render_html


def _scores():
    return [
        ModelScores("opus", 0.9, 0.1, 0.7),
        ModelScores("sonnet", 0.7, 0.4, 0.5),
        ModelScores("haiku", 0.4, 0.8, 0.2),
    ]


def test_html_contains_models_and_structure():
    html = render_html(_scores(), {"solve_vs_calibration_mace": -0.5, "solve_vs_adversarial": 1.0})
    assert "<html" in html.lower()
    for name in ("opus", "sonnet", "haiku"):
        assert name in html
    assert "Dissociation" in html or "dissociation" in html
    assert "%" in html


def test_html_handles_none_correlation():
    html = render_html(_scores(), {"solve_vs_calibration_mace": None, "solve_vs_adversarial": None})
    assert "n/a" in html.lower()
