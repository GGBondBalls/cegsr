from __future__ import annotations

from copy import deepcopy

from cegsr.agents.graph_runtime import AgentGraphRuntime
from cegsr.trajectories.replay import episode_to_markdown
from cegsr.trajectories.schema import TaskSample


class SiriusLiteBaseline:
    """
    Whole-trajectory feedback rewrite baseline:
    keep successful trajectories,
    rewrite the whole failed trajectory using failure feedback,
    without fine-grained credit or selective repair.
    """

    def __init__(self, runtime: AgentGraphRuntime) -> None:
        self.runtime = runtime

    def run(self, sample: TaskSample):
        episode = self.runtime.run_sample(sample, use_retrieval=False, episode_id=f"sirius_lite_{sample.sample_id}")
        if episode.metrics.get("accuracy", 0):
            return episode
        new_sample = deepcopy(sample)
        new_sample.metadata["previous_failure"] = episode_to_markdown(episode, include_gold_answer=False)
        new_episode = self.runtime.run_sample(
            new_sample,
            use_retrieval=False,
            episode_id=f"sirius_lite_retry_{sample.sample_id}",
            extra_context={"rewrite_entire_trajectory": True},
        )
        new_episode.meta["baseline"] = "sirius_lite"
        return new_episode
