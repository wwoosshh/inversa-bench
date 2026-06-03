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


def test_detailed_html_shows_per_item_evidence():
    from inversa.adapters.fake import FakeAdapter
    from inversa.experiment import evaluate_model_detailed
    from inversa.report_html import render_html_detailed
    poser = FakeAdapter(["3", "2*x = 6", "(x - 3)*(x**2 + 1) = 0"])
    weak = FakeAdapter(["999"])
    run = evaluate_model_detailed(
        "fake", poser, weak,
        [{"equation": "2*x + 1 = 7", "answer": 3}], [3], [1], [3],
    )
    out = render_html_detailed([run], {"solve_vs_calibration_mace": None, "solve_vs_adversarial": None})
    assert "per-item evidence" in out.lower()
    assert "fake" in out
    assert "x**2 + 1" in out  # the posed adversarial equation is shown


def test_detailed_html_escapes_model_text():
    # a model output containing '<' must be HTML-escaped, not break the page
    from inversa.adapters.fake import FakeAdapter
    from inversa.experiment import evaluate_model_detailed
    from inversa.report_html import render_html_detailed
    poser = FakeAdapter(["3", "2*x = 6", "x < 3 = 0"])
    weak = FakeAdapter(["1"])
    run = evaluate_model_detailed("m", poser, weak,
                                  [{"equation": "2*x + 1 = 7", "answer": 3}], [3], [1], [3])
    out = render_html_detailed([run], {})
    assert "&lt;" in out  # the '<' was escaped
