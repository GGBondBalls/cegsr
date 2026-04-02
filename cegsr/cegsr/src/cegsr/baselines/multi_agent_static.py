from __future__ import annotations

from cegsr.agents.graph_runtime import AgentGraphRuntime
from cegsr.trajectories.schema import TaskSample


class StaticMultiAgentBaseline:
    def __init__(self, runtime: AgentGraphRuntime) -> None:
        self.runtime = runtime

    def run(self, sample: TaskSample):
        return self.runtime.run_sample(sample, use_retrieval=False, episode_id=f"static_{sample.sample_id}")
