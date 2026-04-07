from __future__ import annotations

from typing import Any

from cegsr.backends.base import BaseBackend, BackendResponse, GenerationConfig
from cegsr.backends.mock_backend import MockBackend
from cegsr.utils.modeling import resolve_local_model_path
from cegsr.utils.logging import get_logger

logger = get_logger(__name__)


class HFLocalBackend(BaseBackend):
    """
    Local HuggingFace backend with chat-template and LoRA adapter support.

    Supports:
    - resolves HF cache repo roots to snapshot directories
    - uses tokenizer.apply_chat_template when available
    - loads per-role LoRA adapters via PEFT and switches at inference time
    - explicit unload() to free GPU memory between inference and training phases
    - gracefully falls back to MockBackend when dependencies or model loading fail
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
        adapter_paths: dict[str, str] | None = None,
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
        self._torch = None
        self._has_adapters = False
        self._adapter_names: list[str] = []

        self._load_model()
        if adapter_paths:
            self._load_adapters(adapter_paths)

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info('Loading model: %s', self.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                device_map=self.device,
                **self.load_kwargs,
            )
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch
            logger.info('Model loaded successfully')
        except Exception as exc:
            logger.warning('Failed to load model, falling back to MockBackend: %s', exc)
            self._tokenizer = None
            self._model = None
            self._torch = None

    def _load_adapters(self, adapter_paths: dict[str, str]) -> None:
        """Load per-role LoRA adapters via PEFT.

        Args:
            adapter_paths: mapping of role_name → adapter directory path.
                           e.g. {"solver": "/path/to/solver_lora", "planner": "/path/to/planner_lora"}
        """
        if self._model is None:
            logger.warning('Cannot load adapters: base model not loaded')
            return
        try:
            from peft import PeftModel
        except ImportError:
            logger.warning('peft not installed, cannot load LoRA adapters')
            return

        first_role = True
        for role, path in adapter_paths.items():
            try:
                if first_role:
                    self._model = PeftModel.from_pretrained(
                        self._model, path, adapter_name=role,
                    )
                    first_role = False
                else:
                    self._model.load_adapter(path, adapter_name=role)
                self._adapter_names.append(role)
                logger.info('Loaded LoRA adapter: %s → %s', role, path)
            except Exception as exc:
                logger.warning('Failed to load adapter %s from %s: %s', role, path, exc)

        if self._adapter_names:
            self._has_adapters = True
            self._model.eval()
            logger.info('All adapters loaded: %s', self._adapter_names)

    def _set_active_adapter(self, role: str | None) -> None:
        """Switch to the adapter for the given role, if available."""
        if not self._has_adapters or role is None:
            return
        if role in self._adapter_names:
            self._model.set_adapter(role)
        # If role not in adapters (e.g. single_agent), keep current adapter active

    def unload(self) -> None:
        """Explicitly free GPU memory. Call between inference and training phases."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._has_adapters = False
        self._adapter_names.clear()
        if self._torch is not None:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
                self._torch.cuda.synchronize()
            logger.info('GPU memory released')
        import gc
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

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

        # Switch LoRA adapter based on the calling role
        if metadata and self._has_adapters:
            self._set_active_adapter(metadata.get('role'))

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
