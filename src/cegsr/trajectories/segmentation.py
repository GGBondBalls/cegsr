from __future__ import annotations

from cegsr.trajectories.schema import EpisodeTrajectory, SubTrajectory


def segment_episode(
    episode: EpisodeTrajectory,
    window_size: int = 2,
    boundary_roles: tuple[str, ...] = ("verifier", "summarizer"),
) -> list[SubTrajectory]:
    """
    Rule-based segmentation:
    1) open a new segment at role boundaries,
    2) cut when the segment reaches the turn window size.
    """
    segments: list[SubTrajectory] = []
    current_turn_ids: list[str] = []
    current_roles: list[str] = []
    current_start = 0

    def flush(end_idx: int) -> None:
        nonlocal current_turn_ids, current_roles, current_start
        if not current_turn_ids:
            return
        summary = " | ".join(f"{t.role}:{t.response[:80]}" for t in episode.turns[current_start : end_idx + 1])
        segments.append(
            SubTrajectory(
                sub_id=f"{episode.episode_id}_sub_{len(segments)}",
                turn_ids=list(current_turn_ids),
                roles=list(current_roles),
                summary=summary,
                start_turn=current_start,
                end_turn=end_idx,
            )
        )
        current_turn_ids = []
        current_roles = []

    for idx, turn in enumerate(episode.turns):
        if not current_turn_ids:
            current_start = idx
        current_turn_ids.append(turn.turn_id)
        current_roles.append(turn.role)
        must_flush = len(current_turn_ids) >= window_size or turn.role in boundary_roles
        if must_flush:
            flush(idx)

    flush(len(episode.turns) - 1)
    episode.subtrajectories = segments
    return segments
