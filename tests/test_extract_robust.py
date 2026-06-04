"""Format-pollution fix: extract_equation must survive LaTeX, '^' caret powers, code
fences and verbose reasoning so weak/verbose models are not scored as false zeros."""
from inversa.tasks.posing import extract_equation
from inversa.verifiers.math_equation import verify_equation


def test_marker_wins_after_reasoning():
    raw = "Let me think. The root should be 3.\nSo:\n#### (x - 3)*(x**2 + 1) = 0"
    assert extract_equation(raw) == "(x - 3)*(x**2 + 1) = 0"


def test_latex_and_caret_are_normalized():
    assert extract_equation("#### $x^2 - 9 = 0$") == "x**2 - 9 = 0"
    assert extract_equation("$$2^x = 8$$") == "2**x = 8"


def test_picks_parseable_line_over_prose():
    raw = "We want the answer.\nFinal: x**2 - 4 = 0"
    eq = extract_equation(raw)
    # the extracted equation must actually verify (root 2), proving prose didn't pollute it
    assert verify_equation(eq, 2).valid is True


def test_code_fence_stripped():
    assert extract_equation("```\nx + 5 = 12\n```") == "x + 5 = 12"


def test_extracted_equation_round_trips_through_verifier():
    raw = "Reasoning blah blah.\n#### x**3 - 27 = 0"
    eq = extract_equation(raw)
    assert verify_equation(eq, 3).unique is True


def test_unicode_superscript_and_minus_normalized():
    from inversa.tasks.posing import extract_equation
    from inversa.verifiers.math_equation import verify_equation
    assert extract_equation("#### x³ - 27 = 0") == "x**3 - 27 = 0"
    # unicode minus (U+2212) and superscript together
    eq = extract_equation("#### x² − 9 = 0")
    assert verify_equation(eq, 3).valid is True


def test_reformat_retry_recovers_buried_equation():
    from inversa.adapters.fake import FakeAdapter
    from inversa.tasks.structural import run_struct_pose_item
    # first reply is prose with no equation; the one-shot reformat retry supplies it
    poser = FakeAdapter(["Let me think about this, the answer is three.", "#### x = 3"])
    item = run_struct_pose_item(poser, "3", 3.0)
    assert item.valid is True
    assert "[reformat-retry]" in item.raw_output
