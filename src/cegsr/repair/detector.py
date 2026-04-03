from __future__ import annotations

from typing import Any

from cegsr.trajectories.schema import EpisodeTrajectory


def detect_low_credit_targets(
    episode: EpisodeTrajectory,
    turn_threshold: float = 0.45,
    subtrajectory_threshold: float = 0.45,
    require_verifier_issue: bool = True,
    verifier_issue_threshold: float = 0.6,
    relax_on_failure: bool = True,
    failure_margin: float = 0.08,
) -> list[dict[str, Any]]:
    """
    Mark spans for selective repair:
    - low fused credit
    - optionally also low verifier signal
    """
    flagged: list[dict[str, Any]] = []
    failed_episode = float(episode.metrics.get("accuracy", 0)) == 0.0
    for record in episode.credit_records:
        verifier_signal = record.signals.get("verifier", 0.5)
        role = str(record.details.get("role", ""))
        effective_turn_threshold = turn_threshold
        if failed_episode and relax_on_failure and role in {"solver", "summarizer"}:
            effective_turn_threshold += failure_margin
        effective_subtrajectory_threshold = subtrajectory_threshold
        if failed_episode and relax_on_failure:
            effective_subtrajectory_threshold += failure_margin
        if record.target_type == "turn" and record.total < effective_turn_threshold:
            threshold = verifier_issue_threshold if failed_episode else 0.5
            if not require_verifier_issue or verifier_signal < threshold:
                flagged.append({"target_type": "turn", "target_id": record.target_id, "credit": record.total})
            elif failed_episode and relax_on_failure and role in {"solver", "summarizer"}:
                flagged.append({"target_type": "turn", "target_id": record.target_id, "credit": record.total})
        elif record.target_type == "subtrajectory" and record.total < effective_subtrajectory_threshold:
            threshold = verifier_issue_threshold if failed_episode else 0.5
            if not require_verifier_issue or verifier_signal < threshold:
                flagged.append({"target_type": "subtrajectory", "target_id": record.target_id, "credit": record.total})
            elif failed_episode and relax_on_failure:
                flagged.append({"target_type": "subtrajectory", "target_id": record.target_id, "credit": record.total})
    return flagged
