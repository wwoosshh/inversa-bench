"""The engine must surface per-model token spend (incl. reasoning tokens + truncations) so
runaway reasoning models and truncation-corrupted scores are visible, not hidden in the bill."""
from __future__ import annotations

from inversa.adapters.openai_compatible import UsageTotals
from inversa.cli_bench import _usage_dict


def test_usage_dict_serializes_all_token_fields():
    u = UsageTotals(calls=3, prompt_tokens=30, completion_tokens=600,
                    reasoning_tokens=540, truncations=2)
    assert _usage_dict(u) == {
        "calls": 3, "prompt_tokens": 30, "completion_tokens": 600,
        "reasoning_tokens": 540, "truncations": 2,
    }


def test_usage_dict_handles_adapters_without_usage():
    assert _usage_dict(None) is None
