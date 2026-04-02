from __future__ import annotations

from pathlib import Path

from cegsr.trajectories.schema import EpisodeTrajectory
from cegsr.utils.io import ensure_dir


def episode_to_markdown(episode: EpisodeTrajectory) -> str:
    """Render one episode into a human-readable markdown trace."""
    lines = [
        f"# Episode {episode.episode_id}",
        "",
        f"**Question**: {episode.sample.question}",
        f"**Final prediction**: {episode.final_prediction}",
        f"**Gold answer**: {episode.sample.answer}",
        f"**Metrics**: {episode.metrics}",
        "",
        "## Turns",
    ]
    for turn in episode.turns:
        lines.extend(
            [
                f"### {turn.turn_id} / {turn.role}",
                "",
                f"Dependencies: {turn.dependencies}",
                "",
                turn.response,
                "",
            ]
        )
    if episode.repair_records:
        lines.append("## Repairs")
        for repair in episode.repair_records:
            lines.append(f"- {repair.repair_id}: {repair.why_repaired}")
    return "\n".join(lines)


def export_episode_markdown(episode: EpisodeTrajectory, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    Path(path).write_text(episode_to_markdown(episode), encoding="utf-8")
