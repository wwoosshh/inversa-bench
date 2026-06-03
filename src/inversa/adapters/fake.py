"""Deterministic fake adapter for tests (no network)."""
from __future__ import annotations

from typing import List, Sequence


class FakeAdapter:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses: List[str] = list(responses)
        self._i = 0
        self.prompts: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise ValueError("FakeAdapter has no responses configured")
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r
