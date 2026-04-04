from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from cegsr.trajectories.schema import EpisodeTrajectory
from cegsr.utils.io import ensure_dir, write_json, write_jsonl
from cegsr.utils.logging import get_logger

logger = get_logger(__name__)


def _build_credit_map(episodes: list[EpisodeTrajectory]) -> dict[tuple[str, str], float]:
    """Build (episode_id, turn_id) → credit score lookup."""
    score_map: dict[tuple[str, str], float] = {}
    for ep in episodes:
        for rec in ep.credit_records:
            if rec.target_type == "turn":
                score_map[(ep.episode_id, rec.target_id)] = rec.total
    return score_map


def _is_repair_success(episode: EpisodeTrajectory) -> bool:
    """Check if repair changed outcome from incorrect to correct."""
    if not episode.repair_records:
        return False
    return episode.metrics.get("accuracy", 0) >= 1.0 and any(
        r.meta.get("outcome_changed") for r in episode.repair_records
    )


def export_role_sft(episodes: list[EpisodeTrajectory], output_dir: str | Path) -> dict[str, str]:
    """
    Export role-specific SFT data in OpenAI/sharegpt-compatible message format
    suitable for LLaMA-Factory custom dataset registration.

    Exports ALL turns (no credit filtering). Use export_credit_guided_sft for filtered export.
    """
    output_dir = ensure_dir(output_dir)
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        for turn in episode.turns:
            record = {
                "messages": turn.prompt_messages + [{"role": "assistant", "content": turn.response}],
                "meta": {
                    "sample_id": episode.sample.sample_id,
                    "episode_id": episode.episode_id,
                    "role": turn.role,
                    "task_type": episode.sample.task_type,
                },
            }
            by_role[turn.role].append(record)
    manifest: dict[str, str] = {}
    for role, rows in by_role.items():
        path = output_dir / f"{role}_sft.jsonl"
        write_jsonl(path, rows)
        manifest[role] = str(path)
    write_json(output_dir / "sft_manifest.json", manifest)
    return manifest


def export_credit_guided_sft(
    episodes: list[EpisodeTrajectory],
    output_dir: str | Path,
    high_credit_threshold: float = 0.65,
    include_repair_success: bool = True,
) -> dict[str, str]:
    """
    Export credit-guided SFT data: only high-quality turns enter training.

    Selection criteria:
    1. Turns with credit >= high_credit_threshold from successful episodes
    2. Turns from episodes where repair changed outcome to correct (if include_repair_success)

    Returns manifest: {role: file_path}.
    """
    output_dir = ensure_dir(output_dir)
    score_map = _build_credit_map(episodes)
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = {"total_turns": 0, "selected_high_credit": 0, "selected_repair_success": 0, "rejected": 0}

    for episode in episodes:
        is_repair_ok = include_repair_success and _is_repair_success(episode)
        ep_correct = episode.metrics.get("accuracy", 0) >= 1.0

        for turn in episode.turns:
            stats["total_turns"] += 1
            credit = score_map.get((episode.episode_id, turn.turn_id), 0.0)
            source = None

            if ep_correct and credit >= high_credit_threshold:
                source = "high_credit"
                stats["selected_high_credit"] += 1
            elif is_repair_ok:
                source = "repair_success"
                stats["selected_repair_success"] += 1
            else:
                stats["rejected"] += 1
                continue

            record = {
                "messages": turn.prompt_messages + [{"role": "assistant", "content": turn.response}],
                "meta": {
                    "sample_id": episode.sample.sample_id,
                    "episode_id": episode.episode_id,
                    "role": turn.role,
                    "task_type": episode.sample.task_type,
                    "credit": round(credit, 4),
                    "source": source,
                },
            }
            by_role[turn.role].append(record)

    manifest: dict[str, str] = {}
    for role, rows in by_role.items():
        path = output_dir / f"{role}_sft.jsonl"
        write_jsonl(path, rows)
        manifest[role] = str(path)

    write_json(output_dir / "sft_manifest.json", manifest)
    write_json(output_dir / "credit_filter_stats.json", stats)
    logger.info(
        "Credit-guided export: %d/%d turns selected (high_credit=%d, repair_success=%d, rejected=%d)",
        stats["selected_high_credit"] + stats["selected_repair_success"],
        stats["total_turns"],
        stats["selected_high_credit"],
        stats["selected_repair_success"],
        stats["rejected"],
    )
    return manifest


def export_preference_pairs(episodes: list[EpisodeTrajectory], output_dir: str | Path) -> str:
    """Create chosen/rejected pairs from selective repairs."""
    output_dir = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for repair in episode.repair_records:
            old_text = "\n".join(item.get("response", "") for item in repair.old_span)
            new_text = "\n".join(item.get("response", "") for item in repair.new_span)
            if not old_text.strip() or not new_text.strip():
                continue
            prompt_messages = []
            if episode.turns:
                prompt_messages = episode.turns[0].prompt_messages
            rows.append(
                {
                    "messages": prompt_messages,
                    "chosen": {"role": "assistant", "content": new_text},
                    "rejected": {"role": "assistant", "content": old_text},
                    "meta": {"episode_id": episode.episode_id, "repair_id": repair.repair_id},
                }
            )
    path = output_dir / "preference_pairs.jsonl"
    write_jsonl(path, rows)
    return str(path)


def export_reward_data(episodes: list[EpisodeTrajectory], output_dir: str | Path) -> str:
    """
    Placeholder reward/KTO-style export.
    We use a simple binary kto_tag from fused turn credit.
    """
    output_dir = ensure_dir(output_dir)
    score_map = {}
    for episode in episodes:
        for record in episode.credit_records:
            score_map[(episode.episode_id, record.target_id)] = record.total
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for turn in episode.turns:
            score = score_map.get((episode.episode_id, turn.turn_id), 0.0)
            rows.append(
                {
                    "messages": turn.prompt_messages + [{"role": "assistant", "content": turn.response}],
                    "kto_tag": bool(score >= 0.5),
                    "reward": score,
                    "meta": {"episode_id": episode.episode_id, "turn_id": turn.turn_id, "role": turn.role},
                }
            )
    path = output_dir / "reward_data.jsonl"
    write_jsonl(path, rows)
    return str(path)
