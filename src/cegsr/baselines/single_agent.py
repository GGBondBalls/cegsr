from __future__ import annotations

from copy import deepcopy

from cegsr.agents.graph_runtime import AgentGraphRuntime
from cegsr.trajectories.schema import TaskSample


class SingleAgentBaseline:
    def __init__(self, runtime: AgentGraphRuntime) -> None:
        self.runtime = runtime

    def run(self, sample: TaskSample):
        original = deepcopy(self.runtime.role_order)
        self.runtime.role_order = ["single_agent"] if "single_agent" in self.runtime.agents else [original[1] if len(original) > 1 else original[0]]
        episode = self.runtime.run_sample(sample, use_retrieval=False, episode_id=f"single_{sample.sample_id}")
        self.runtime.role_order = original
        return episode
