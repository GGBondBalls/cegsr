from __future__ import annotations

from typing import Any

from cegsr.backends.base import BaseBackend, BackendResponse, GenerationConfig
from cegsr.backends.mock_backend import MockBackend
from cegsr.utils.modeling import resolve_local_model_path


class HFLocalBackend(BaseBackend):
    """
    Local HuggingFace backend with chat-template support.

    Behavior:
    - resolves HF cache repo roots to snapshot directories
    - uses tokenizer.apply_chat_template when available
    - gracefully falls back to MockBackend when local dependencies or model loading fail
    """

    backend_name = 'hf_local'

    def __init__(
        self,
        model_name_or_path: str,
        device: str = 'auto',
        trust_remote_code: bool = True,
        load_kwargs: dict[str, Any] | None = None,
        use_chat_template: bool = True,
        model_size: str | None = None,
    ) -> None:
        self.original_model_name_or_path = model_name_or_path
        self.model_name_or_path = resolve_local_model_path(model_name_or_path, model_size_hint=model_size)
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.load_kwargs = load_kwargs or {}
        self.use_chat_template = use_chat_template
        self._tokenizer = None
        self._model = None
        self._fallback = MockBackend()

        try:  # pragma: no cover - optional heavy dependency
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=trust_remote_code,
                device_map=device,
                **self.load_kwargs,
            )
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch
        except Exception:
            self._tokenizer = None
            self._model = None
            self._torch = None

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        if self._tokenizer is not None and self.use_chat_template and hasattr(self._tokenizer, 'apply_chat_template'):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return '\n'.join(f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages) + '\nASSISTANT:'

    def generate(
        self,
        messages: list[dict[str, str]],
        generation_config: GenerationConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BackendResponse:
        cfg = generation_config or GenerationConfig()
        if self._tokenizer is None or self._model is None or self._torch is None:
            return self._fallback.generate(messages, cfg, metadata)

        prompt = self._messages_to_prompt(messages)
        inputs = self._tokenizer([prompt], return_tensors='pt')
        model_device = getattr(self._model, 'device', None)
        if model_device is not None and hasattr(inputs, 'to'):
            inputs = inputs.to(model_device)
        do_sample = cfg.temperature > 0
        generate_kwargs: dict[str, Any] = {
            'max_new_tokens': cfg.max_tokens,
            'do_sample': do_sample,
            'pad_token_id': getattr(self._tokenizer, 'pad_token_id', None) or getattr(self._tokenizer, 'eos_token_id', None),
        }
        if do_sample:
            generate_kwargs['temperature'] = max(cfg.temperature, 1e-5)
            generate_kwargs['top_p'] = cfg.top_p
        with self._torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                **generate_kwargs,
            )
        generated = outputs[0][inputs['input_ids'].shape[-1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return BackendResponse(
            text=text,
            raw={'backend': 'hf_local', 'model_name_or_path': self.model_name_or_path},
            input_tokens=int(inputs['input_ids'].shape[-1]),
            output_tokens=int(generated.shape[-1]) if hasattr(generated, 'shape') else max(1, len(text.split())),
        )
