from __future__ import annotations

from pathlib import Path

from cegsr.trajectories.schema import EpisodeTrajectory
from cegsr.utils.io import append_jsonl, ensure_dir


class TrajectoryRecorder:
    """Persist episode trajectories to disk as JSONL."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        ensure_dir(self.output_path.parent)

    def record(self, episode: EpisodeTrajectory) -> None:
        append_jsonl(self.output_path, episode.to_dict())
