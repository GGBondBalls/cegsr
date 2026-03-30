from __future__ import annotations

from cegsr.credit.base import CreditSignal
from cegsr.trajectories.schema import CreditRecord, EpisodeTrajectory


class OutcomeCreditSignal(CreditSignal):
    """
    Outcome-based heuristic credit:
    - if episode succeeds, all turns receive positive credit,
    - later turns receive a slightly larger score,
    - failed episodes yield lower global credit.
    """

    name = "outcome"

    def compute(self, episode: EpisodeTrajectory) -> list[CreditRecord]:
        success = float(episode.metrics.get("accuracy", 0))
        total_turns = max(1, len(episode.turns))
        records: list[CreditRecord] = []
        for idx, turn in enumerate(episode.turns):
            local_bonus = 0.1 * ((idx + 1) / total_turns)
            score = 0.2 + 0.7 * success + local_bonus
            if success == 0 and idx < total_turns - 1:
                score -= 0.1
            score = max(0.0, min(1.0, score))
            records.append(
                CreditRecord(
                    target_type="turn",
                    target_id=turn.turn_id,
                    total=score,
                    signals={self.name: score},
                    details={"success": success, "position": idx},
                )
            )
        for sub in episode.subtrajectories:
            covered = [r.total for r in records if r.target_id in set(sub.turn_ids)]
            score = sum(covered) / max(1, len(covered))
            records.append(
                CreditRecord(
                    target_type="subtrajectory",
                    target_id=sub.sub_id,
                    total=score,
                    signals={self.name: score},
                    details={"turn_ids": sub.turn_ids},
                )
            )
        role_scores: dict[str, list[float]] = {}
        for turn_record, turn in zip([r for r in records if r.target_type == "turn"], episode.turns):
            role_scores.setdefault(turn.role, []).append(turn_record.total)
        for role, scores in role_scores.items():
            score = sum(scores) / len(scores)
            records.append(
                CreditRecord(
                    target_type="role",
                    target_id=role,
                    total=score,
                    signals={self.name: score},
                    details={"num_turns": len(scores)},
                )
            )
        return records
