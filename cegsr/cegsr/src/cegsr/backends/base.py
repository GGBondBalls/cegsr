from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 256
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)
    json_mode: bool = False


@dataclass
class BackendResponse:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"


class BaseBackend:
    """Unified model backend interface."""

    backend_name = "base"

    def generate(
        self,
        messages: list[dict[str, str]],
        generation_config: GenerationConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BackendResponse:
        raise NotImplementedError

    def batch_generate(
        self,
        batch_messages: list[list[dict[str, str]]],
        generation_config: GenerationConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[BackendResponse]:
        return [self.generate(m, generation_config=generation_config, metadata=metadata) for m in batch_messages]

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(max(1, len(m.get("content", "").split())) for m in messages)
