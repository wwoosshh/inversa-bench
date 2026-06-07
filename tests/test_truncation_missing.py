"""Truncation-as-missing (rebuttal #10): a reasoning model that exhausts its token budget before
emitting the '#### eq' line produces empty content -> currently scored as a WRONG answer (false
zero), deflating exactly the strongest models. With this fix a truncated item is treated as MISSING
(not reached) and excluded from the validity-rate denominator, so the score reflects only items the
model actually attempted. Opt-in; the missing count is reported separately.
"""
from __future__ import annotations

from inversa.adapters.openai_compatible import OpenAICompatibleAdapter
from inversa.bench_core import evaluate_leaderboard
from inversa.tasks.structural import run_struct_pose_item


# --- adapter exposes per-call truncation -----------------------------------

class _FakeResp:
    def __init__(self, content, finish):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content}),
                                        "finish_reason": finish})]
        self.usage = None


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp
        self.chat = type("Ch", (), {"completions": type("Co", (), {"create": lambda s, **k: resp})()})()


def _adapter(content, finish):
    ad = OpenAICompatibleAdapter("m", base_url="http://x", api_key="k")
    ad._client = _FakeClient(_FakeResp(content, finish))
    return ad


def test_adapter_marks_truncated_call():
    ad = _adapter("", "length")          # empty content + length finish = truncation
    ad.generate("p")
    assert ad.last_truncated is True


def test_adapter_clears_truncated_on_good_call():
    ad = _adapter("#### x = 3", "stop")
    ad.generate("p")
    assert ad.last_truncated is False


# --- runner marks a truncated, answerless item as not-answered --------------

class _TruncAdapter:
    last_truncated = True
    def generate(self, prompt):          # truncated model: no content at all
        return ""


class _NormalFake:
    last_truncated = False
    def __init__(self, eq):
        self._eq = eq
    def generate(self, prompt):
        return self._eq


def test_truncated_item_is_not_answered():
    it = run_struct_pose_item(_TruncAdapter(), "3", 3.0)
    assert it.answered is False           # truncated + no parseable equation -> missing

def test_real_attempt_is_answered_even_if_wrong():
    it = run_struct_pose_item(_NormalFake("#### x**2 - 9 = 0"), "3", 3.0)
    assert it.answered is True            # produced an equation (wrong/non-unique) -> a real attempt


# --- engine excludes missing (None) items from the rate ---------------------

def _items(n):
    return [{"i": k} for k in range(n)]


def test_engine_excludes_missing_items_from_denominator():
    pose, trans = _items(10), _items(10)

    def scorer(model, bank, item, rep):
        if bank == "pose" and item["i"] % 2 == 0:
            return None                   # 5 pose items missing (truncated)
        return True                       # the rest answered & valid

    res = evaluate_leaderboard(["a"], pose, trans, scorer, repeats=1, max_workers=4, adaptive=False)
    # 5 missing excluded -> pose rate is 5/5 = 1.0, NOT 5/10 = 0.5
    assert res[0]["pose_validity"] == 1.0
    assert res[0]["transform_validity"] == 1.0


def test_engine_skips_model_with_everything_missing():
    pose, trans = _items(6), _items(6)
    res = evaluate_leaderboard(["ghost"], pose, trans, lambda *a: None,
                               repeats=1, max_workers=4, adaptive=False)
    assert res == []                      # all-missing model can't be scored -> skipped
