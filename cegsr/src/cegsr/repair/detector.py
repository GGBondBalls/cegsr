from __future__ import annotations

from typing import Any

from cegsr.trajectories.schema import EpisodeTrajectory


def detect_low_credit_targets(
    episode: EpisodeTrajectory,
    turn_threshold: float = 0.45,
    subtrajectory_threshold: float = 0.45,
    require_verifier_issue: bool = True,
) -> list[dict[str, Any]]:
    """
    Mark spans for selective repair:
    - low fused credit
    - optionally also low verifier signal
    """
    flagged: list[dict[str, Any]] = []
    for record in episode.credit_records:
        verifier_signal = record.signals.get("verifier", 0.5)
        if record.target_type == "turn" and record.total < turn_threshold:
            if not require_verifier_issue or verifier_signal < 0.5:
                flagged.append({"target_type": "turn", "target_id": record.target_id, "credit": record.total})
        elif record.target_type == "subtrajectory" and record.total < subtrajectory_threshold:
            if not require_verifier_issue or verifier_signal < 0.5:
                flagged.append({"target_type": "subtrajectory", "target_id": record.target_id, "credit": record.total})
    return flagged
