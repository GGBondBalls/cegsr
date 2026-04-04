from __future__ import annotations

from pathlib import Path
from typing import Any

from cegsr.utils.io import ensure_dir, write_json, write_yaml
from cegsr.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default training templates (Qwen2.5 + dual 4090 optimized)
# ---------------------------------------------------------------------------

DEFAULT_LORA_TEMPLATE: dict[str, Any] = {
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "template": "qwen",
    "cutoff_len": 2048,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "num_train_epochs": 3.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 500,
    "plot_loss": True,
    "overwrite_output_dir": True,
}

DEFAULT_QLORA_TEMPLATE: dict[str, Any] = {
    **DEFAULT_LORA_TEMPLATE,
    "quantization_bit": 4,
    "quantization_method": "bnb",
    "double_quantization": True,
}

DEFAULT_DPO_TEMPLATE: dict[str, Any] = {
    "stage": "dpo",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "pref_beta": 0.1,
    "pref_loss": "sigmoid",
    "template": "qwen",
    "cutoff_len": 2048,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-6,
    "num_train_epochs": 1.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 500,
    "plot_loss": True,
    "overwrite_output_dir": True,
}

MERGE_TEMPLATE: dict[str, Any] = {
    "template": "qwen",
    "finetuning_type": "lora",
    "export_size": 2,
    "export_device": "cpu",
    "export_legacy_format": False,
}


def build_dataset_info(
    manifest: dict[str, str],
    preference_path: str | None = None,
    reward_path: str | None = None,
) -> dict[str, Any]:
    """Build LLaMA-Factory dataset_info.json from export manifest."""
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
    # Combined dataset (all roles merged) for single-adapter training
    all_role_files = [Path(p).name for p in manifest.values()]
    if len(all_role_files) > 1:
        for i, (role, path) in enumerate(manifest.items()):
            info[f"all_roles_part{i}"] = {
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
    if preference_path and Path(preference_path).exists():
        info["repair_preference"] = {
            "file_name": Path(preference_path).name,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "messages",
                "chosen": "chosen",
                "rejected": "rejected",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    if reward_path and Path(reward_path).exists():
        info["turn_reward"] = {
            "file_name": Path(reward_path).name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "kto_tag": "kto_tag"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    return info


# ---------------------------------------------------------------------------
# Shell script builders
# ---------------------------------------------------------------------------

def _normalize_gpu_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _build_single_process_script(command_lines: list[str]) -> str:
    return "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", "", *command_lines]) + "\n"


def _build_distributed_script(command_lines: list[str], distributed_config: dict[str, Any]) -> str:
    gpu_list = _normalize_gpu_list(distributed_config.get("gpus"))
    env_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if gpu_list:
        env_lines.append(f'export CUDA_VISIBLE_DEVICES="{",".join(gpu_list)}"')
    if distributed_config.get("force_torchrun", True):
        env_lines.append("export FORCE_TORCHRUN=1")
    env_lines.append(f'export NNODES="{distributed_config.get("nnodes", 1)}"')
    env_lines.append(f'export NODE_RANK="{distributed_config.get("node_rank", 0)}"')
    env_lines.append(f'export NPROC_PER_NODE="{distributed_config.get("nproc_per_node", max(1, len(gpu_list)) or 1)}"')
    env_lines.append(f'export MASTER_ADDR="{distributed_config.get("master_addr", "127.0.0.1")}"')
    env_lines.append(f'export MASTER_PORT="{distributed_config.get("master_port", 29500)}"')
    env_lines.append("")
    return "\n".join([*env_lines, *command_lines]) + "\n"


# ---------------------------------------------------------------------------
# Config generation for SFT, DPO, merge
# ---------------------------------------------------------------------------

def _generate_sft_configs(
    export_dir: Path,
    manifest: dict[str, str],
    model_name_or_path: str,
    output_base: str,
    lora_template: dict[str, Any],
    qlora_template: dict[str, Any],
) -> tuple[dict[str, str], list[str]]:
    """Generate per-role SFT configs. Returns (config_paths, command_lines)."""
    config_paths: dict[str, str] = {}
    command_lines: list[str] = []

    # Per-role SFT configs
    for role in manifest:
        role_name = f"{role}_sft"
        for mode_name, template in [("lora", lora_template), ("qlora", qlora_template)]:
            cfg = {
                **template,
                "model_name_or_path": model_name_or_path,
                "dataset_dir": str(export_dir),
                "dataset": role_name,
                "output_dir": str(Path(output_base) / f"{role}_{mode_name}"),
            }
            cfg_path = export_dir / f"llamafactory_{role}_{mode_name}.yaml"
            write_yaml(cfg_path, cfg)
            command_lines.append(f"llamafactory-cli train {cfg_path}")
            config_paths[f"{role}_{mode_name}"] = str(cfg_path)

    # Combined (all roles) SFT config
    all_datasets = ",".join(f"all_roles_part{i}" for i in range(len(manifest)))
    if len(manifest) > 1:
        for mode_name, template in [("lora", lora_template), ("qlora", qlora_template)]:
            cfg = {
                **template,
                "model_name_or_path": model_name_or_path,
                "dataset_dir": str(export_dir),
                "dataset": all_datasets,
                "output_dir": str(Path(output_base) / f"combined_{mode_name}"),
            }
            cfg_path = export_dir / f"llamafactory_combined_{mode_name}.yaml"
            write_yaml(cfg_path, cfg)
            config_paths[f"combined_{mode_name}"] = str(cfg_path)

    return config_paths, command_lines


def _generate_dpo_configs(
    export_dir: Path,
    model_name_or_path: str,
    output_base: str,
    sft_adapter_path: str | None,
    dpo_template: dict[str, Any],
) -> tuple[dict[str, str], list[str]]:
    """Generate DPO config (uses repair preference pairs). Returns (config_paths, command_lines)."""
    config_paths: dict[str, str] = {}
    command_lines: list[str] = []

    preference_file = export_dir / "preference_pairs.jsonl"
    if not preference_file.exists():
        return config_paths, command_lines

    cfg: dict[str, Any] = {
        **dpo_template,
        "model_name_or_path": model_name_or_path,
        "dataset_dir": str(export_dir),
        "dataset": "repair_preference",
        "output_dir": str(Path(output_base) / "dpo"),
    }
    if sft_adapter_path:
        cfg["adapter_name_or_path"] = sft_adapter_path

    cfg_path = export_dir / "llamafactory_dpo.yaml"
    write_yaml(cfg_path, cfg)
    command_lines.append(f"llamafactory-cli train {cfg_path}")
    config_paths["dpo"] = str(cfg_path)
    return config_paths, command_lines


def _generate_merge_config(
    export_dir: Path,
    model_name_or_path: str,
    adapter_path: str,
    merged_output_dir: str,
    merge_template: dict[str, Any] | None = None,
) -> str:
    """Generate config for merging LoRA adapter into base model."""
    template = merge_template or MERGE_TEMPLATE
    cfg = {
        **template,
        "model_name_or_path": model_name_or_path,
        "adapter_name_or_path": adapter_path,
        "export_dir": merged_output_dir,
    }
    cfg_path = export_dir / "llamafactory_merge.yaml"
    write_yaml(cfg_path, cfg)
    return str(cfg_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_llamafactory_project(
    export_dir: str,
    model_name_or_path: str,
    output_dir: str,
    lora_template: dict[str, Any] | None = None,
    qlora_template: dict[str, Any] | None = None,
    dpo_template: dict[str, Any] | None = None,
    distributed_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Generate dataset_info.json, per-role YAML configs, DPO config, merge config,
    and helper shell scripts for LLaMA-Factory training.
    """
    import json

    export_dir_path = ensure_dir(export_dir)
    dataset_manifest = export_dir_path / "sft_manifest.json"
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))

    preference_path = str(export_dir_path / "preference_pairs.jsonl") if (export_dir_path / "preference_pairs.jsonl").exists() else None
    reward_path = str(export_dir_path / "reward_data.jsonl") if (export_dir_path / "reward_data.jsonl").exists() else None
    dataset_info = build_dataset_info(manifest, preference_path=preference_path, reward_path=reward_path)
    write_json(export_dir_path / "dataset_info.json", dataset_info)

    lora_template = {**DEFAULT_LORA_TEMPLATE, **(lora_template or {})}
    qlora_template = {**DEFAULT_QLORA_TEMPLATE, **(qlora_template or {})}
    dpo_template = {**DEFAULT_DPO_TEMPLATE, **(dpo_template or {})}

    # SFT configs
    sft_paths, sft_commands = _generate_sft_configs(
        export_dir_path, manifest, model_name_or_path, output_dir,
        lora_template, qlora_template,
    )

    # DPO config (chains after combined SFT adapter)
    combined_sft_adapter = str(Path(output_dir) / "combined_lora")
    dpo_paths, dpo_commands = _generate_dpo_configs(
        export_dir_path, model_name_or_path, output_dir,
        sft_adapter_path=combined_sft_adapter,
        dpo_template=dpo_template,
    )

    # Merge config (merge final adapter into base model)
    final_adapter = str(Path(output_dir) / "dpo") if dpo_paths else combined_sft_adapter
    merged_output = str(Path(output_dir) / "merged_model")
    merge_cfg_path = _generate_merge_config(
        export_dir_path, model_name_or_path, final_adapter, merged_output,
    )

    config_paths = {**sft_paths, **dpo_paths, "merge": merge_cfg_path}

    # Shell scripts
    all_commands = sft_commands + dpo_commands
    (export_dir_path / "run_llamafactory.sh").write_text(
        _build_single_process_script(all_commands), encoding="utf-8",
    )
    if distributed_config:
        (export_dir_path / "run_llamafactory_ddp.sh").write_text(
            _build_distributed_script(all_commands, distributed_config), encoding="utf-8",
        )

    # Merge script
    merge_commands = [f"llamafactory-cli export {merge_cfg_path}"]
    (export_dir_path / "run_merge.sh").write_text(
        _build_single_process_script(merge_commands), encoding="utf-8",
    )

    logger.info("LLaMA-Factory project generated: %d configs in %s", len(config_paths), export_dir)
    return config_paths
