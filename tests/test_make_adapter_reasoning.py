"""The engine must thread a reasoning-token budget down to the OpenAI-compatible adapter,
otherwise the CLI flag that bounds reasoning-model spend is silently dropped."""
from __future__ import annotations

from inversa.cli_experiment import _make_adapter


def test_make_adapter_forwards_reasoning_budget(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    ad = _make_adapter("some/model", "openrouter", "http://x", "OPENROUTER_API_KEY",
                       max_tokens=2048, reasoning_max_tokens=4000)
    assert ad._reasoning_max_tokens == 4000


def test_make_adapter_defaults_reasoning_budget_to_none(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    ad = _make_adapter("some/model", "openrouter", "http://x", "OPENROUTER_API_KEY",
                       max_tokens=2048)
    assert ad._reasoning_max_tokens is None


def test_make_adapter_forwards_temperature(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    ad = _make_adapter("some/model", "openrouter", "http://x", "OPENROUTER_API_KEY",
                       max_tokens=2048, temperature=0.0)
    assert ad._temperature == 0.0
