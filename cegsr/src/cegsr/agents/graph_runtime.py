from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from cegsr.agents.base import BaseAgent
from cegsr.trajectories.schema import EpisodeTrajectory, TaskSample
from cegsr.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentGraphRuntime:
    agents: dict[str, BaseAgent]
    role_order: list[str]
    task: Any
    retriever: Any | None = None

    def _log_turn(self, turn: Any, verbose_turns: bool) -> None:
        if not verbose_turns:
            return
        logger.info("[%s|%s]\n%s", turn.turn_id, turn.role, turn.response)

    def _extract_final_prediction(self, episode: EpisodeTrajectory) -> str:
        if not episode.turns:
            return ""
        for turn in reversed(episode.turns):
            prediction = self.task.extract_prediction(turn.response)
            if prediction:
                return prediction
        return episode.turns[-1].response

    def run_sample(
        self,
        sample: TaskSample,
        use_retrieval: bool = False,
        episode_id: str | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> EpisodeTrajectory:
        started = time.time()
        episode = EpisodeTrajectory(
            episode_id=episode_id or f"ep_{sample.sample_id}",
            sample=sample,
            turns=[],
        )
        extra_context = extra_context or {}
        verbose_turns = bool(extra_context.get("verbose_turns", False))
        enabled_roles = extra_context.get("retrieval_enabled_roles")
        for idx, role in enumerate(self.role_order):
            agent = self.agents[role]
            retrieved = []
            if use_retrieval and self.retriever is not None and (not enabled_roles or role in enabled_roles):
                retrieved = self.retriever.retrieve(
                    role=role,
                    task_type=sample.task_type,
                    query=sample.question,
                    history=episode.turns,
                    sample_id=sample.sample_id,
                    dataset_name=sample.metadata.get("dataset_name"),
                    top_k=extra_context.get("top_k"),
                )
            turn = agent.act(
                sample=sample,
                task=self.task,
                history=episode.turns,
                retrieved_experience=retrieved,
                extra_context=extra_context,
                turn_id=f"{episode.episode_id}_t{idx}_{role}",
            )
            turn.meta["retrieved_node_ids"] = [n.node_id for n in retrieved]
            episode.turns.append(turn)
            self._log_turn(turn, verbose_turns)
        episode.final_prediction = self._extract_final_prediction(episode)
        episode.metrics = self.task.evaluate_prediction(sample, episode.final_prediction)
        episode.reward = float(episode.metrics.get("accuracy", 0))
        episode.input_tokens = sum(t.input_tokens for t in episode.turns)
        episode.output_tokens = sum(t.output_tokens for t in episode.turns)
        episode.latency_s = time.time() - started
        return episode

    def rerun_suffix(
        self,
        sample: TaskSample,
        prefix_turns: list[Any],
        start_index: int,
        use_retrieval: bool = False,
        extra_context: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Re-run only the suffix turns from a chosen turn index."""
        history = list(prefix_turns)
        extra_context = extra_context or {}
        verbose_turns = bool(extra_context.get("verbose_turns", False))
        enabled_roles = extra_context.get("retrieval_enabled_roles")
        for idx in range(start_index, len(self.role_order)):
            role = self.role_order[idx]
            agent = self.agents[role]
            retrieved = []
            if use_retrieval and self.retriever is not None and (not enabled_roles or role in enabled_roles):
                retrieved = self.retriever.retrieve(
                    role=role,
                    task_type=sample.task_type,
                    query=sample.question,
                    history=history,
                    sample_id=sample.sample_id,
                    dataset_name=sample.metadata.get("dataset_name"),
                    top_k=extra_context.get("top_k"),
                )
            turn = agent.act(
                sample=sample,
                task=self.task,
                history=history,
                retrieved_experience=retrieved,
                extra_context=extra_context,
                turn_id=f"rerun_{sample.sample_id}_{idx}_{role}",
            )
            history.append(turn)
            self._log_turn(turn, verbose_turns)
        return history
