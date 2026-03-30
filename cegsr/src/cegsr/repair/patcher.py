from __future__ import annotations

from copy import deepcopy

from cegsr.trajectories.schema import EpisodeTrajectory, RepairRecord


def apply_repair_patch(
    episode: EpisodeTrajectory,
    turn_index: int,
    new_turns: list,
    repair_record: RepairRecord,
) -> EpisodeTrajectory:
    """Replace one turn and its dependent suffix with new turns."""
    updated = deepcopy(episode)
    updated.turns = updated.turns[:turn_index] + new_turns
    updated.repair_records.append(repair_record)
    return updated
