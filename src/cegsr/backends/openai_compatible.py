from __future__ import annotations

import time
from typing import Any

import requests

from cegsr.backends.base import BaseBackend, BackendResponse, GenerationConfig
from cegsr.utils.logging import get_logger

logger = get_logger(__name__)


class ServerDownError(ConnectionError):
    """The inference server is unreachable (crashed or not started)."""


class OpenAICompatibleBackend(BaseBackend):
    """Use local OpenAI-compatible inference servers such as vLLM or SGLang."""

    backend_name = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: int = 120,
        max_retries: int = 3,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
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

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
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

            except requests.exceptions.ConnectionError as exc:
                # Server is down — no point retrying without a restart
                raise ServerDownError(
                    f"Inference server unreachable at {self.base_url}: {exc}"
                ) from exc

            except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "Request timeout (attempt %d/%d), retrying in %ds ...",
                        attempt, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    # Check if server is still alive before retrying
                    if not self._server_alive():
                        raise ServerDownError(
                            f"Inference server died at {self.base_url}"
                        ) from exc

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status >= 500 and attempt < self.max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "Server error %d (attempt %d/%d), retrying in %ds ...",
                        status, attempt, self.max_retries, wait,
                    )
                    time.sleep(wait)
                else:
                    raise

        raise last_exc  # type: ignore[misc]

    def _server_alive(self) -> bool:
        try:
            resp = self.session.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False
