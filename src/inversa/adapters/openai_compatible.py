"""Adapter for any OpenAI Chat Completions-compatible endpoint (OpenRouter, Ollama,
OpenAI, ...). Same `Adapter` interface (generate) as the rest of Inversa, so the
posing/calibration/adversarial/solve runners work unchanged across model families."""
from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


class OpenAICompatibleAdapter:
    def __init__(self, model: str, base_url: str, api_key: Optional[str] = None,
                 api_key_env: Optional[str] = None, max_tokens: int = 512) -> None:
        key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self._client = OpenAI(base_url=base_url, api_key=key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content
        return content or ""
