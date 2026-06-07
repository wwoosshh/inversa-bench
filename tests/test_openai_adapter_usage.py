"""Adapter behavior that prevents the reasoning-model failure modes diagnosed on OpenRouter:

  Mode A (provider IGNORES max_tokens, e.g. deepseek-r1): reasoning runs uncapped ->
    3-4x the configured token spend, billed in full. Fix: a reasoning-token budget the
    adapter forwards, plus per-call usage accounting so runaway spend is visible.
  Mode B (provider HONORS max_tokens, e.g. minimax-m1): reasoning fills the budget ->
    finish_reason='length', empty content, no '#### eq' -> score collapses to a truncation
    artifact. Fix: count truncations so they are distinguishable from genuine wrong answers.
"""
from __future__ import annotations

from types import SimpleNamespace

from inversa.adapters.openai_compatible import OpenAICompatibleAdapter


def _resp(content, finish_reason="stop", completion=0, reasoning=0, prompt=0):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(
            completion_tokens=completion,
            prompt_tokens=prompt,
            total_tokens=completion + prompt,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
    )


class _FakeClient:
    """Records create() kwargs and returns a scripted response (no network)."""

    def __init__(self, response):
        self._response = response
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _adapter(response, **kw):
    ad = OpenAICompatibleAdapter("some/model", base_url="http://x", api_key="k", **kw)
    ad._client = _FakeClient(response)
    return ad


def test_forwards_reasoning_budget_in_extra_body():
    ad = _adapter(_resp("#### x = 3"), max_tokens=2048, reasoning_max_tokens=4000)
    ad.generate("p")
    sent = ad._client.calls[0]
    assert sent["extra_body"]["reasoning"]["max_tokens"] == 4000


def test_no_reasoning_extra_body_when_budget_unset():
    ad = _adapter(_resp("#### x = 3"), max_tokens=2048)
    ad.generate("p")
    sent = ad._client.calls[0]
    assert "reasoning" not in (sent.get("extra_body") or {})


def test_generate_accumulates_token_usage():
    ad = _adapter(_resp("#### x = 3", completion=120, reasoning=90, prompt=15))
    ad.generate("p")
    assert ad.usage.calls == 1
    assert ad.usage.completion_tokens == 120
    assert ad.usage.reasoning_tokens == 90
    assert ad.usage.prompt_tokens == 15


def test_counts_truncation_on_length_finish_with_empty_content():
    ad = _adapter(_resp("", finish_reason="length", completion=2048, reasoning=2048))
    out = ad.generate("p")
    assert out == ""
    assert ad.usage.truncations == 1


def test_no_truncation_counted_on_normal_stop():
    ad = _adapter(_resp("#### x = 3", finish_reason="stop", completion=50))
    ad.generate("p")
    assert ad.usage.truncations == 0


def test_generate_still_returns_content():
    ad = _adapter(_resp("#### x = 3"))
    assert ad.generate("p") == "#### x = 3"


def test_forwards_temperature_when_set():
    # temperature=0 collapses run-to-run sampling noise, so the leaderboard needs far fewer
    # repeats to separate models — the biggest single cost lever after capping reasoning.
    ad = _adapter(_resp("#### x = 3"), temperature=0.0)
    ad.generate("p")
    assert ad._client.calls[0]["temperature"] == 0.0


def test_omits_temperature_when_unset():
    ad = _adapter(_resp("#### x = 3"))
    ad.generate("p")
    assert "temperature" not in ad._client.calls[0]
