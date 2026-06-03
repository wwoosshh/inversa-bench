"""Adapter protocol: anything that turns a prompt into a text completion."""
from __future__ import annotations

from typing import Protocol


class Adapter(Protocol):
    def generate(self, prompt: str) -> str:
        """Return the model's text completion for `prompt`."""
        ...
