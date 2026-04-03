from __future__ import annotations

from cegsr.backends.openai_compatible import OpenAICompatibleBackend


class SGLangBackend(OpenAICompatibleBackend):
    backend_name = "sglang"
