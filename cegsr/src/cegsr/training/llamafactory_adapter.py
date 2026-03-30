from __future__ import annotations

from pathlib import Path
from typing import Any

from cegsr.utils.io import ensure_dir, write_json, write_yaml


def build_dataset_info(manifest: dict[str, str], preference_path: str | None = None, reward_path: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for role, path in manifest.items():
        info[f"{role}_sft"] = {
            "file_name": Path(path).name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    if preference_path:
        info["repair_preference"] = {
            "file_name": Path(preference_path).name,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {"messages": "messages", "chosen": "chosen", "rejected": "rejected"},
        }
    if reward_path:
        info["turn_reward"] = {
            "file_name": Path(reward_path).name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "kto_tag": "kto_tag"},
        }
    return info


def generate_llamafactory_project(
    export_dir: str,
    model_name_or_path: str,
    output_dir: str,
    lora_template: dict[str, Any] | None = None,
    qlora_template: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Generate dataset_info.json, per-role YAML configs and helper shell scripts.
    """
    export_dir_path = ensure_dir(export_dir)
    dataset_manifest = Path(export_dir_path / "sft_manifest.json")
    import json

    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    preference_path = str(export_dir_path / "preference_pairs.jsonl") if (export_dir_path / "preference_pairs.jsonl").exists() else None
    reward_path = str(export_dir_path / "reward_data.jsonl") if (export_dir_path / "reward_data.jsonl").exists() else None
    dataset_info = build_dataset_info(manifest, preference_path=preference_path, reward_path=reward_path)
    write_json(export_dir_path / "dataset_info.json", dataset_info)

    lora_template = lora_template or {
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "num_train_epochs": 1.0,
        "cutoff_len": 2048,
        "template": "default",
    }
    qlora_template = qlora_template or {
        **lora_template,
        "quantization_bit": 4,
        "double_quantization": True,
    }

    config_paths: dict[str, str] = {}
    command_lines: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for role in manifest:
        role_name = f"{role}_sft"
        for mode_name, template in [("lora", lora_template), ("qlora", qlora_template)]:
            cfg = {
                **template,
                "model_name_or_path": model_name_or_path,
                "dataset_dir": str(export_dir_path),
                "dataset": role_name,
                "output_dir": str(Path(output_dir) / f"{role}_{mode_name}"),
            }
            cfg_path = export_dir_path / f"llamafactory_{role}_{mode_name}.yaml"
            write_yaml(cfg_path, cfg)
            command_lines.append(f"llamafactory-cli train {cfg_path}")
            config_paths[f"{role}_{mode_name}"] = str(cfg_path)

    (export_dir_path / "run_llamafactory.sh").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    return config_paths
