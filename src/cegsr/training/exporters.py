from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from cegsr.trajectories.schema import AgentTurn, EpisodeTrajectory
from cegsr.utils.io import ensure_dir, write_json, write_jsonl


def export_role_sft(episodes: list[EpisodeTrajectory], output_dir: str | Path) -> dict[str, str]:
    """
    Export role-specific SFT data in OpenAI/sharegpt-compatible message format
    suitable for LLaMA-Factory custom dataset registration.
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
    filtered_turns: list[tuple[EpisodeTrajectory, AgentTurn]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Export credit-filtered role-specific SFT data.

    Unlike export_role_sft which exports ALL turns, this only exports
    turns that passed credit-guided filtering.
    """
    output_dir = ensure_dir(output_dir)
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode, turn in filtered_turns:
        record = {
            "messages": turn.prompt_messages + [{"role": "assistant", "content": turn.response}],
            "meta": {
                "sample_id": episode.sample.sample_id,
                "episode_id": episode.episode_id,
                "role": turn.role,
                "task_type": episode.sample.task_type,
                "credit_guided": True,
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


def export_credit_guided_preference(
    preference_pairs: list[dict[str, Any]],
    output_dir: str | Path,
) -> str:
    """Export credit-guided preference pairs for DPO training."""
    output_dir = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for pair in preference_pairs:
        rows.append({
            "messages": pair["prompt_messages"],
            "chosen": {"role": "assistant", "content": pair["chosen"]},
            "rejected": {"role": "assistant", "content": pair["rejected"]},
            "meta": {
                "episode_id": pair.get("episode_id"),
                "repair_id": pair.get("repair_id"),
            },
        })
    path = output_dir / "preference_pairs.jsonl"
    write_jsonl(path, rows)
    return str(path)


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
