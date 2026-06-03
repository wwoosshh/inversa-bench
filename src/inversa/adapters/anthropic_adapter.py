"""Real model adapter backed by the Anthropic API."""
from __future__ import annotations

import os

from anthropic import Anthropic


class AnthropicAdapter:
    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None,
                 max_tokens: int = 256) -> None:
        self._client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")
