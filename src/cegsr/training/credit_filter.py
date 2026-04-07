"""Credit-guided training data selection.

Filters episode turns by credit scores to produce high-quality training data
instead of blindly using all turns as SiriuS does.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from cegsr.trajectories.schema import AgentTurn, CreditRecord, EpisodeTrajectory


@dataclass
class CreditFilterConfig:
    high_threshold: float = 0.65
    low_threshold: float = 0.35
    include_repaired: bool = True
    min_samples_per_role: int = 10
    fallback_threshold_step: float = 0.05


@dataclass
class FilteredData:
    """Container for credit-filtered training data."""
    sft_turns: list[tuple[EpisodeTrajectory, AgentTurn]] = field(default_factory=list)
    preference_pairs: list[dict[str, Any]] = field(default_factory=list)
    negative_turns: list[tuple[EpisodeTrajectory, AgentTurn]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _build_credit_map(episode: EpisodeTrajectory) -> dict[str, float]:
    """Map turn_id → fused credit score."""
    return {
        r.target_id: r.total
        for r in episode.credit_records
        if r.target_type == "turn"
    }


def _is_repair_success(episode: EpisodeTrajectory, turn: AgentTurn) -> bool:
    """Check if this turn was repaired and the episode became correct after repair."""
    if not episode.repair_records:
        return False
    repaired_turn_ids = set()
    for rec in episode.repair_records:
        for span_item in rec.new_span:
            tid = span_item.get("turn_id", "")
            if tid:
                repaired_turn_ids.add(tid)
    if turn.turn_id not in repaired_turn_ids:
        return False
    return episode.metrics.get("accuracy", 0) >= 0.5


def filter_by_credit(
    episodes: list[EpisodeTrajectory],
    config: CreditFilterConfig | None = None,
) -> FilteredData:
    """Filter episodes to produce credit-guided training data.

    Returns:
        FilteredData with SFT turns, preference pairs, and negative turns.
    """
    cfg = config or CreditFilterConfig()
    result = FilteredData()

    role_counts: dict[str, int] = defaultdict(int)
    total_turns = 0
    high_credit_count = 0
    repaired_count = 0

    for episode in episodes:
        credit_map = _build_credit_map(episode)

        for turn in episode.turns:
            total_turns += 1
            score = credit_map.get(turn.turn_id, 0.0)

            # High credit turns → SFT positive
            if score >= cfg.high_threshold:
                result.sft_turns.append((episode, turn))
                role_counts[turn.role] += 1
                high_credit_count += 1
                continue

            # Successfully repaired turns → SFT positive
            if cfg.include_repaired and _is_repair_success(episode, turn):
                result.sft_turns.append((episode, turn))
                role_counts[turn.role] += 1
                repaired_count += 1
                continue

            # Low credit turns → KTO negative signal
            if score < cfg.low_threshold:
                result.negative_turns.append((episode, turn))

    # Fallback: if any role has too few samples, lower threshold
    all_roles = sorted({t.role for ep in episodes for t in ep.turns})
    for role in all_roles:
        if role_counts[role] >= cfg.min_samples_per_role:
            continue
        threshold = cfg.high_threshold - cfg.fallback_threshold_step
        while threshold >= cfg.low_threshold and role_counts[role] < cfg.min_samples_per_role:
            for episode in episodes:
                credit_map = _build_credit_map(episode)
                for turn in episode.turns:
                    if turn.role != role:
                        continue
                    score = credit_map.get(turn.turn_id, 0.0)
                    already_in = any(t.turn_id == turn.turn_id for _, t in result.sft_turns)
                    if not already_in and score >= threshold:
                        result.sft_turns.append((episode, turn))
                        role_counts[role] += 1
            threshold -= cfg.fallback_threshold_step

    # Extract preference pairs from repair records
    for episode in episodes:
        for repair in episode.repair_records:
            old_text = "\n".join(item.get("response", "") for item in repair.old_span)
            new_text = "\n".join(item.get("response", "") for item in repair.new_span)
            if not old_text.strip() or not new_text.strip():
                continue
            if old_text.strip() == new_text.strip():
                continue
            prompt_messages = []
            if episode.turns:
                prompt_messages = episode.turns[0].prompt_messages
            result.preference_pairs.append({
                "episode_id": episode.episode_id,
                "repair_id": repair.repair_id,
                "role": repair.target_id.split("_")[-1] if "_" in repair.target_id else "",
                "prompt_messages": prompt_messages,
                "chosen": new_text,
                "rejected": old_text,
            })

    result.stats = {
        "total_turns": total_turns,
        "sft_selected": len(result.sft_turns),
        "sft_high_credit": high_credit_count,
        "sft_repaired": repaired_count,
        "preference_pairs": len(result.preference_pairs),
        "negative_turns": len(result.negative_turns),
        "per_role": dict(role_counts),
        "selection_rate": len(result.sft_turns) / max(1, total_turns),
    }
    return result
