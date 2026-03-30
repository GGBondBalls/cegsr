from __future__ import annotations

from typing import Any

from cegsr.agents.base import BaseAgent
from cegsr.agents.prompts import DEFAULT_ROLE_PROMPTS
from cegsr.backends.base import GenerationConfig


def build_agents(agent_configs: list[dict[str, Any]], backend: Any) -> dict[str, BaseAgent]:
    """Instantiate role agents from config."""
    agents: dict[str, BaseAgent] = {}
    for cfg in agent_configs:
        role = cfg["role"]
        agents[role] = BaseAgent(
            role=role,
            backend=backend,
            system_prompt=cfg.get("system_prompt", DEFAULT_ROLE_PROMPTS.get(role, f"You are {role}.")),
            generation_config=GenerationConfig(**cfg.get("generation_config", {})),
        )
    return agents
