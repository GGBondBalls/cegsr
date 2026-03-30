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

    def _verdict(self, text: str) -> str | None:
        lowered = text.lower()
        if 'verdict:' in lowered:
            verdict_match = re.search(r'VERDICT:\s*(correct|incorrect)', text, re.I)
            if verdict_match:
                return verdict_match.group(1).lower()
        if 'incorrect' in lowered or 'issue' in lowered or 'mistake' in lowered:
            return 'incorrect'
        if 'correct' in lowered:
            return 'correct'
        return None

    def _score_text(self, text: str) -> float | None:
        score_match = re.search(r'Score:\s*([01](?:\.\d+)?)', text, re.I)
        if score_match:
            score = max(0.0, min(1.0, float(score_match.group(1))))
            verdict = self._verdict(text)
            if verdict == 'correct':
                return min(score, 0.85)
            if verdict == 'incorrect':
                return min(score, 0.35)
            return score
        verdict = self._verdict(text)
        if verdict == 'correct':
            return 0.9
        if verdict == 'incorrect':
            return 0.25
        return None

    def compute(self, episode: EpisodeTrajectory) -> list[CreditRecord]:
        verifier_turns = [t for t in episode.turns if t.role == 'verifier']
        verifier_scores = [self._score_text(t.response) for t in verifier_turns]
        verifier_scores = [x for x in verifier_scores if x is not None]
        verifier_verdicts = [self._verdict(t.response) for t in verifier_turns]
        verifier_verdicts = [x for x in verifier_verdicts if x is not None]
        default_score = verifier_scores[-1] if verifier_scores else 0.5
        default_verdict = verifier_verdicts[-1] if verifier_verdicts else None
        records: list[CreditRecord] = []
        for turn in episode.turns:
            if turn.role == 'verifier':
                score = self._score_text(turn.response) or default_score
            else:
                score = 0.55
                if default_verdict == 'correct':
                    score += 0.1
                elif default_verdict == 'incorrect':
                    score -= 0.15
                if turn.role == 'solver':
                    score += 0.05
                elif turn.role == 'planner':
                    score -= 0.05
                elif turn.role == 'summarizer':
                    score += 0.05 if episode.metrics.get('accuracy', 0) else -0.15
                if episode.sample.answer and episode.sample.answer.lower() in turn.response.lower():
                    score = max(score, 0.72 if default_verdict != 'incorrect' else 0.45)
                if 'not sure' in turn.response.lower() or 'maybe' in turn.response.lower():
                    score = min(score, 0.45)
                if 'answer:' in turn.response.lower() and episode.metrics.get('accuracy', 0) == 0:
                    score = min(score, 0.42)
                score = max(0.0, min(1.0, score))
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
