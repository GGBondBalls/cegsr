from __future__ import annotations

import time
from typing import Any

import requests

from cegsr.backends.base import BaseBackend, BackendResponse, GenerationConfig


class OpenAICompatibleBackend(BaseBackend):
    """Use local OpenAI-compatible inference servers such as vLLM or SGLang."""

    backend_name = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: int = 120,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.session = requests.Session()

    def generate(
        self,
        messages: list[dict[str, str]],
        generation_config: GenerationConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BackendResponse:
        cfg = generation_config or GenerationConfig()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
        }
        if cfg.stop:
            payload["stop"] = cfg.stop
        payload.update(self.extra_body)
        started = time.time()
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        latency = time.time() - started
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return BackendResponse(
            text=choice,
            raw={"latency_s": latency, "response": data},
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
        )
