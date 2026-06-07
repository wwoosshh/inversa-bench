"""Adapter for any OpenAI Chat Completions-compatible endpoint (OpenRouter, Ollama,
OpenAI, ...). Same `Adapter` interface (generate) as the rest of Inversa, so the
posing/calibration/adversarial/solve runners work unchanged across model families.

Reasoning models on OpenRouter break a naive `max_tokens`-only call two ways (both measured):
  - providers that IGNORE max_tokens let reasoning run uncapped -> 3-4x the configured spend;
    `reasoning_max_tokens` forwards a reasoning budget so cost is bounded.
  - providers that HONOR max_tokens let reasoning fill the budget -> finish_reason='length',
    empty content, no '#### eq' -> a truncation that must NOT be read as a wrong answer.
`usage` accumulates per-call token spend + a truncation count so both modes are observable.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class UsageTotals:
    """Running token-spend totals for one adapter (one model), so the engine can report
    real cost and flag runaway reasoning instead of only seeing the upstream bill."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    truncations: int = 0  # finish_reason == 'length' with no usable content


class OpenAICompatibleAdapter:
    def __init__(self, model: str, base_url: str, api_key: Optional[str] = None,
                 api_key_env: Optional[str] = None, max_tokens: int = 512,
                 reasoning_max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> None:
        key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        # max_retries: SDK retries 429/5xx/timeout with exponential backoff, so a
        # transient upstream rate-limit doesn't crash a long multi-model experiment.
        self._client = OpenAI(base_url=base_url, api_key=key, max_retries=3, timeout=45.0)
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_max_tokens = reasoning_max_tokens
        self._temperature = temperature
        # one adapter is shared across a model's concurrently-scheduled items (engine A), so
        # the usage tally is mutated from several threads — guard it to keep totals exact.
        self._usage_lock = threading.Lock()
        self.usage = UsageTotals()
        self.last_truncated = False  # did the most recent generate() truncate (budget exhausted,
        #                              empty content)? lets the runner treat it as missing, not wrong.

    def generate(self, prompt: str) -> str:
        kwargs = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if self._temperature is not None:
            # temperature=0 makes scoring near-deterministic, so run-to-run noise (the reason
            # repeats are needed) shrinks — fewer repeats reach the same separation. Sent only
            # when set so providers that reject the param on reasoning models aren't disturbed.
            kwargs["temperature"] = self._temperature
        if self._reasoning_max_tokens is not None:
            # OpenRouter forwards this to providers that support a reasoning budget; on
            # providers that ignore max_tokens it is the only thing that bounds spend.
            kwargs["extra_body"] = {"reasoning": {"max_tokens": self._reasoning_max_tokens}}
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content or ""
        self._record_usage(resp, choice.finish_reason, content)
        return content

    def _record_usage(self, resp, finish_reason, content: str) -> None:
        u = getattr(resp, "usage", None)
        self.last_truncated = finish_reason == "length" and not content.strip()
        with self._usage_lock:
            self.usage.calls += 1
            if u is not None:
                self.usage.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
                self.usage.completion_tokens += getattr(u, "completion_tokens", 0) or 0
                details = getattr(u, "completion_tokens_details", None)
                self.usage.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0
            if finish_reason == "length" and not content.strip():
                self.usage.truncations += 1
