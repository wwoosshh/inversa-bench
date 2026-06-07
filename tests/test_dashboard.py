"""The leaderboard dashboard must summarize a (possibly PARTIAL) IGS run honestly: how many of
the attempted roster were actually measured, which families they span, the IGS spread, token
spend, and — crucially for a paper-grade artifact — which models went unmeasured. The summary is
a pure function so the synthesis is tested without a browser or a live run.
"""
from __future__ import annotations

from inversa.dashboard import summarize_leaderboard


def _row(model, igs, pose=None, transform=None, reps=1, usage=None):
    return {"model": model, "igs": igs,
            "pose_validity": pose if pose is not None else igs,
            "transform_validity": transform if transform is not None else igs,
            "reps_done": reps, "usage": usage}


def test_summary_counts_measured_vs_attempted_and_lists_unmeasured():
    roster = ["openai/gpt-5", "openai/gpt-4o-mini", "x-ai/grok-4.3", "minimax/minimax-m1"]
    results = [_row("openai/gpt-5", 1.0), _row("openai/gpt-4o-mini", 0.6)]
    s = summarize_leaderboard(results, roster)
    assert s["n_attempted"] == 4
    assert s["n_measured"] == 2
    assert s["unmeasured"] == ["x-ai/grok-4.3", "minimax/minimax-m1"]


def test_summary_aggregates_by_family():
    roster = ["openai/gpt-5", "openai/gpt-4o-mini", "anthropic/claude-opus-4.8"]
    results = [_row("openai/gpt-5", 1.0), _row("openai/gpt-4o-mini", 0.6),
               _row("anthropic/claude-opus-4.8", 0.97)]
    s = summarize_leaderboard(results, roster)
    fams = {f["family"]: f for f in s["families"]}
    assert fams["openai"]["n"] == 2
    assert fams["openai"]["best_igs"] == 1.0
    assert fams["openai"]["best_model"] == "openai/gpt-5"
    assert fams["anthropic"]["n"] == 1
    # families ordered by best_igs desc
    assert s["families"][0]["family"] == "openai"


def test_summary_reports_igs_range_and_token_spend():
    roster = ["a/x", "a/y"]
    results = [_row("a/x", 0.9, usage={"completion_tokens": 100, "reasoning_tokens": 40,
                                        "prompt_tokens": 10, "truncations": 0}),
               _row("a/y", 0.2, usage={"completion_tokens": 50, "reasoning_tokens": 0,
                                       "prompt_tokens": 5, "truncations": 2})]
    s = summarize_leaderboard(results, roster)
    assert s["igs_min"] == 0.2 and s["igs_max"] == 0.9
    assert s["total_completion_tokens"] == 150
    assert s["total_reasoning_tokens"] == 40
    assert s["truncated_models"] == ["a/y"]


def test_summary_handles_empty_results():
    s = summarize_leaderboard([], ["a/x", "b/y"])
    assert s["n_measured"] == 0
    assert s["n_attempted"] == 2
    assert s["families"] == []
    assert s["igs_min"] is None and s["igs_max"] is None
