from __future__ import annotations

import re
from typing import Any

from cegsr.trajectories.schema import AgentTurn, TaskSample


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class BaseTask:
    task_type = "base"

    def build_prompt(
        self,
        sample: TaskSample,
        role: str,
        retrieved_experience: list[Any],
        history: list[AgentTurn],
        system_prompt: str,
        extra_context: dict[str, Any],
    ) -> list[dict[str, str]]:
        raise NotImplementedError

    def extract_prediction(self, text: str) -> str:
        raise NotImplementedError

    def evaluate_prediction(self, sample: TaskSample, prediction: str) -> dict[str, Any]:
        gold = normalize_text(sample.answer)
        pred = normalize_text(prediction)
        exact = int(gold == pred)
        return {"accuracy": exact, "exact_match": exact}
