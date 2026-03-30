from __future__ import annotations

import re

from cegsr.credit.base import CreditSignal
from cegsr.trajectories.schema import CreditRecord, EpisodeTrajectory


class VerifierCreditSignal(CreditSignal):
    """
    Extract local process scores from verifier turns when available,
    otherwise back off to answer-consistency heuristics.
    """

    name = 'verifier'

    def _score_text(self, text: str) -> float | None:
        score_match = re.search(r'Score:\s*([01](?:\.\d+)?)', text, re.I)
        if score_match:
            return max(0.0, min(1.0, float(score_match.group(1))))
        lowered = text.lower()
        if 'correct' in lowered and 'incorrect' not in lowered:
            return 0.9
        if 'incorrect' in lowered or 'issue' in lowered or 'mistake' in lowered:
            return 0.25
        return None

    def compute(self, episode: EpisodeTrajectory) -> list[CreditRecord]:
        verifier_scores = [self._score_text(t.response) for t in episode.turns if t.role == 'verifier']
        verifier_scores = [x for x in verifier_scores if x is not None]
        default_score = max(verifier_scores) if verifier_scores else 0.5
        records: list[CreditRecord] = []
        for turn in episode.turns:
            if turn.role == 'verifier':
                score = self._score_text(turn.response) or default_score
            else:
                score = default_score
                if episode.sample.answer and episode.sample.answer.lower() in turn.response.lower():
                    score = max(score, 0.72)
                if 'not sure' in turn.response.lower() or 'maybe' in turn.response.lower():
                    score = min(score, 0.45)
            records.append(
                CreditRecord(
                    target_type='turn',
                    target_id=turn.turn_id,
                    total=score,
                    signals={self.name: score},
                    details={'role': turn.role},
                )
            )
        for sub in episode.subtrajectories:
            sub_scores = [r.total for r in records if r.target_id in set(sub.turn_ids)]
            avg = sum(sub_scores) / max(1, len(sub_scores))
            records.append(
                CreditRecord(
                    target_type='subtrajectory',
                    target_id=sub.sub_id,
                    total=avg,
                    signals={self.name: avg},
                    details={'verifier_proxy': True},
                )
            )
        role_buckets: dict[str, list[float]] = {}
        for turn, record in zip(episode.turns, [r for r in records if r.target_type == 'turn']):
            role_buckets.setdefault(turn.role, []).append(record.total)
        for role, values in role_buckets.items():
            score = sum(values) / len(values)
            records.append(
                CreditRecord(
                    target_type='role',
                    target_id=role,
                    total=score,
                    signals={self.name: score},
                    details={'num_turns': len(values)},
                )
            )
        return records
