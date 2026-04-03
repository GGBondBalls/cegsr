from __future__ import annotations

from copy import deepcopy

from cegsr.credit.verifier_credit import VerifierCreditSignal
from cegsr.repair.detector import detect_low_credit_targets
from cegsr.repair.patcher import apply_repair_patch
from cegsr.trajectories.schema import EpisodeTrajectory, RepairRecord


class SelectiveRepairEngine:
    """
    Local selective repair:
    1) detect low-credit turn/subtrajectory,
    2) repair only the earliest flagged turn,
    3) preserve high-credit prefix as anchors,
    4) re-run the downstream dependent suffix.
    """

    def __init__(
        self,
        runtime,
        turn_threshold: float = 0.45,
        subtrajectory_threshold: float = 0.45,
        require_verifier_issue: bool = True,
        verifier_issue_threshold: float = 0.6,
        relax_on_failure: bool = True,
        failure_margin: float = 0.08,
    ) -> None:
        self.runtime = runtime
        self.turn_threshold = turn_threshold
        self.subtrajectory_threshold = subtrajectory_threshold
        self.require_verifier_issue = require_verifier_issue
        self.verifier_issue_threshold = verifier_issue_threshold
        self.relax_on_failure = relax_on_failure
        self.failure_margin = failure_margin
        self.verifier = VerifierCreditSignal()

    def repair(self, episode: EpisodeTrajectory, use_retrieval: bool = False) -> EpisodeTrajectory:
        flagged = detect_low_credit_targets(
            episode,
            turn_threshold=self.turn_threshold,
            subtrajectory_threshold=self.subtrajectory_threshold,
            require_verifier_issue=self.require_verifier_issue,
            verifier_issue_threshold=self.verifier_issue_threshold,
            relax_on_failure=self.relax_on_failure,
            failure_margin=self.failure_margin,
        )
        if not flagged:
            return episode
        first = flagged[0]
        if first['target_type'] == 'subtrajectory':
            sub = next(s for s in episode.subtrajectories if s.sub_id == first['target_id'])
            target_turn_id = sub.turn_ids[0]
        else:
            target_turn_id = first['target_id']
        target_index = next(i for i, t in enumerate(episode.turns) if t.turn_id == target_turn_id)

        prefix = deepcopy(episode.turns[:target_index])
        original_suffix = deepcopy(episode.turns[target_index:])
        repair_hint = {
            'repair_mode': True,
            'repair_reason': f"low credit={first['credit']:.3f}; selectively rewrite only local problematic span",
            'preserved_context': [t.response for t in prefix[-2:]],
        }
        rerun_turns = self.runtime.rerun_suffix(
            sample=episode.sample,
            prefix_turns=prefix,
            start_index=target_index,
            use_retrieval=use_retrieval,
            extra_context=repair_hint,
        )
        new_suffix = rerun_turns[target_index:]
        verifier_before = next((r.total for r in episode.credit_records if r.target_id == target_turn_id), None)
        updated = apply_repair_patch(
            episode=episode,
            turn_index=target_index,
            new_turns=new_suffix,
            repair_record=RepairRecord(
                repair_id=f'{episode.episode_id}_repair_0',
                target_type='turn',
                target_id=target_turn_id,
                old_span=[t.to_dict() for t in original_suffix],
                new_span=[t.to_dict() for t in new_suffix],
                why_repaired=repair_hint['repair_reason'],
                kept_context_turn_ids=[t.turn_id for t in prefix],
                verifier_before=verifier_before,
            ),
        )
        updated.final_prediction = self.runtime._extract_final_prediction(updated)
        updated.metrics = self.runtime.task.evaluate_prediction(updated.sample, updated.final_prediction)
        updated.reward = float(updated.metrics.get('accuracy', 0))
        updated.input_tokens = sum(t.input_tokens for t in updated.turns)
        updated.output_tokens = sum(t.output_tokens for t in updated.turns)
        verifier_after_records = self.verifier.compute(updated)
        verifier_after = next((r.total for r in verifier_after_records if r.target_id == target_turn_id), None)
        if updated.repair_records:
            updated.repair_records[-1].verifier_after = verifier_after
            updated.repair_records[-1].meta['changed'] = original_suffix != new_suffix
            updated.repair_records[-1].meta['post_repair_accuracy'] = updated.metrics.get('accuracy', 0)
        return updated
