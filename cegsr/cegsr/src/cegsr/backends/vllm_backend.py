from __future__ import annotations

from cegsr.backends.openai_compatible import OpenAICompatibleBackend


class VLLMBackend(OpenAICompatibleBackend):
    backend_name = "vllm"
