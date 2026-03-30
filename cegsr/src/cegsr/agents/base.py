from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cegsr.backends.base import BaseBackend, GenerationConfig
from cegsr.trajectories.schema import AgentTurn, TaskSample


@dataclass
class BaseAgent:
    role: str
    backend: BaseBackend
    system_prompt: str
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)

    def act(
        self,
        sample: TaskSample,
        task: Any,
        history: list[AgentTurn],
        retrieved_experience: list[Any] | None = None,
        extra_context: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> AgentTurn:
        """Run one agent turn."""
        messages = task.build_prompt(
            sample=sample,
            role=self.role,
            retrieved_experience=retrieved_experience or [],
            history=history,
            system_prompt=self.system_prompt,
            extra_context=extra_context or {},
        )
        started = time.time()
        response = self.backend.generate(
            messages=messages,
            generation_config=self.generation_config,
            metadata={
                "role": self.role,
                "sample_id": sample.sample_id,
                "gold_answer": sample.answer,
            },
        )
        latency = time.time() - started
        dependencies = []
        if history:
            dependencies = [history[-1].turn_id]
            if self.role == "summarizer":
                dependencies = [turn.turn_id for turn in history]
        return AgentTurn(
            turn_id=turn_id or f"{sample.sample_id}_{self.role}_{len(history)}",
            role=self.role,
            prompt_messages=messages,
            response=response.text,
            dependencies=dependencies,
            latency_s=latency,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            meta={"raw_backend": response.raw, "finish_reason": response.finish_reason},
        )
