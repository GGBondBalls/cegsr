from __future__ import annotations

from collections import Counter

from cegsr.credit.base import CreditSignal
from cegsr.trajectories.schema import CreditRecord, EpisodeTrajectory


class DependencyCreditSignal(CreditSignal):
    """
    Credit by downstream usage:
    if later turns explicitly depend on a turn, its score is boosted.
    """

    name = "dependency"

    def compute(self, episode: EpisodeTrajectory) -> list[CreditRecord]:
        dependency_counter: Counter[str] = Counter()
        for turn in episode.turns:
            for dep in turn.dependencies:
                dependency_counter[dep] += 1
        max_count = max(dependency_counter.values()) if dependency_counter else 1
        records: list[CreditRecord] = []
        for turn in episode.turns:
            count = dependency_counter.get(turn.turn_id, 0)
            score = min(1.0, 0.2 + 0.8 * (count / max_count if max_count else 0.0))
            records.append(
                CreditRecord(
                    target_type="turn",
                    target_id=turn.turn_id,
                    total=score,
                    signals={self.name: score},
                    details={"downstream_uses": count},
                )
            )
        for sub in episode.subtrajectories:
            sub_scores = [r.total for r in records if r.target_id in set(sub.turn_ids)]
            score = sum(sub_scores) / max(1, len(sub_scores))
            records.append(
                CreditRecord(
                    target_type="subtrajectory",
                    target_id=sub.sub_id,
                    total=score,
                    signals={self.name: score},
                    details={"turn_ids": sub.turn_ids},
                )
            )
        role_bucket: dict[str, list[float]] = {}
        for turn, record in zip(episode.turns, [r for r in records if r.target_type == "turn"]):
            role_bucket.setdefault(turn.role, []).append(record.total)
        for role, values in role_bucket.items():
            score = sum(values) / len(values)
            records.append(CreditRecord(target_type="role", target_id=role, total=score, signals={self.name: score}))
        return records
