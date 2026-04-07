"""LLaMA-Factory training executor.

Calls `llamafactory-cli train <yaml>` as a subprocess, captures logs,
and manages per-role SFT + optional DPO training.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from cegsr.utils.io import ensure_dir, read_json, write_json, write_yaml
from cegsr.utils.logging import get_logger

logger = get_logger(__name__)


def _find_llamafactory_cli() -> str:
    """Locate the llamafactory-cli executable."""
    import shutil
    cli = shutil.which('llamafactory-cli')
    if cli:
        return cli
    # Fallback: try python -m llamafactory
    return f'{sys.executable} -m llamafactory'


def run_llamafactory_train(config_path: str | Path, log_file: str | Path | None = None) -> int:
    """Execute a single LLaMA-Factory training run.

    Args:
        config_path: Path to the YAML config file.
        log_file: Optional path to write training logs.

    Returns:
        Process return code (0 = success).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Training config not found: {config_path}')

    cli = _find_llamafactory_cli()
    cmd = f'{cli} train {config_path}'
    logger.info('Starting training: %s', cmd)

    log_handle = None
    if log_file:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        log_handle = open(log_file, 'w', encoding='utf-8')

    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            stdout=log_handle or sys.stdout,
            stderr=subprocess.STDOUT,
            timeout=3600 * 12,  # 12 hour max
        )
        logger.info('Training finished with return code: %d', proc.returncode)
        return proc.returncode
    except subprocess.TimeoutExpired:
        logger.error('Training timed out after 12 hours')
        return -1
    finally:
        if log_handle:
            log_handle.close()


def generate_sft_config(
    model_name_or_path: str,
    dataset_dir: str,
    dataset_name: str,
    output_dir: str,
    template: str = 'qwen',
    finetuning_type: str = 'lora',
    quantization_bit: int | None = 4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_target: str = 'all',
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    num_train_epochs: float = 2.0,
    cutoff_len: int = 2048,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = 'cosine',
    logging_steps: int = 10,
    save_strategy: str = 'epoch',
    bf16: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a LLaMA-Factory SFT YAML config dict."""
    cfg: dict[str, Any] = {
        'model_name_or_path': model_name_or_path,
        'stage': 'sft',
        'do_train': True,
        'finetuning_type': finetuning_type,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'lora_target': lora_target,
        'template': template,
        'dataset_dir': dataset_dir,
        'dataset': dataset_name,
        'output_dir': output_dir,
        'cutoff_len': cutoff_len,
        'per_device_train_batch_size': per_device_train_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'learning_rate': learning_rate,
        'num_train_epochs': num_train_epochs,
        'lr_scheduler_type': lr_scheduler_type,
        'warmup_ratio': warmup_ratio,
        'logging_steps': logging_steps,
        'save_strategy': save_strategy,
        'bf16': bf16,
        'report_to': 'none',
        'overwrite_output_dir': True,
    }
    if quantization_bit is not None:
        cfg['quantization_bit'] = quantization_bit
        cfg['quantization_method'] = 'bitsandbytes'
    if extra:
        cfg.update(extra)
    return cfg


def generate_dpo_config(
    model_name_or_path: str,
    adapter_name_or_path: str,
    dataset_dir: str,
    dataset_name: str,
    output_dir: str,
    template: str = 'qwen',
    finetuning_type: str = 'lora',
    quantization_bit: int | None = 4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_target: str = 'all',
    dpo_beta: float = 0.1,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 1.0,
    cutoff_len: int = 2048,
    bf16: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a LLaMA-Factory DPO YAML config dict."""
    cfg: dict[str, Any] = {
        'model_name_or_path': model_name_or_path,
        'adapter_name_or_path': adapter_name_or_path,
        'stage': 'dpo',
        'do_train': True,
        'finetuning_type': finetuning_type,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'lora_target': lora_target,
        'template': template,
        'dataset_dir': dataset_dir,
        'dataset': dataset_name,
        'output_dir': output_dir,
        'cutoff_len': cutoff_len,
        'per_device_train_batch_size': per_device_train_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'learning_rate': learning_rate,
        'num_train_epochs': num_train_epochs,
        'dpo_beta': dpo_beta,
        'bf16': bf16,
        'report_to': 'none',
        'overwrite_output_dir': True,
    }
    if quantization_bit is not None:
        cfg['quantization_bit'] = quantization_bit
        cfg['quantization_method'] = 'bitsandbytes'
    if extra:
        cfg.update(extra)
    return cfg


def train_role_sft(
    config: dict[str, Any],
    export_dir: str,
    adapters_dir: str,
    roles: list[str] | None = None,
    training_mode: str = 'qlora',
) -> dict[str, str]:
    """Run per-role SFT training via LLaMA-Factory.

    Args:
        config: Full project config dict.
        export_dir: Directory containing credit-guided exported data + dataset_info.json.
        adapters_dir: Output directory for trained adapters.
        roles: Which roles to train. None = all roles found in manifest.
        training_mode: 'lora' or 'qlora'.

    Returns:
        Dict mapping role → adapter output path.
    """
    from cegsr.utils.modeling import resolve_local_model_path

    export_path = Path(export_dir)
    manifest_file = export_path / 'sft_manifest.json'
    if not manifest_file.exists():
        raise FileNotFoundError(f'SFT manifest not found: {manifest_file}')
    manifest = read_json(manifest_file)

    training_cfg = config.get('training', {})
    model_path = resolve_local_model_path(
        training_cfg.get('model_name_or_path', config['backend']['model_name_or_path']),
        model_size_hint=training_cfg.get('model_size', config['backend'].get('model_size')),
    )

    # Merge user-provided template overrides
    template_key = f'{training_mode}_template'
    user_template = training_cfg.get(template_key, {})

    if roles is None:
        roles = list(manifest.keys())

    adapter_paths: dict[str, str] = {}
    adapters_base = ensure_dir(adapters_dir)

    for role in roles:
        dataset_name = f'{role}_sft'
        adapter_out = str(adapters_base / f'{role}_{training_mode}')

        sft_cfg = generate_sft_config(
            model_name_or_path=model_path,
            dataset_dir=str(export_path),
            dataset_name=dataset_name,
            output_dir=adapter_out,
            template=user_template.get('template', 'qwen'),
            finetuning_type='lora',
            quantization_bit=user_template.get('quantization_bit', 4 if training_mode == 'qlora' else None),
            lora_rank=user_template.get('lora_rank', 16),
            lora_alpha=user_template.get('lora_alpha', 32),
            lora_target=user_template.get('lora_target', 'all'),
            per_device_train_batch_size=user_template.get('per_device_train_batch_size', 2),
            gradient_accumulation_steps=user_template.get('gradient_accumulation_steps', 8),
            learning_rate=user_template.get('learning_rate', 2e-4),
            num_train_epochs=user_template.get('num_train_epochs', 2.0),
            cutoff_len=user_template.get('cutoff_len', 2048),
            bf16=user_template.get('bf16', True),
        )
        cfg_path = export_path / f'train_{role}_{training_mode}.yaml'
        write_yaml(cfg_path, sft_cfg)

        logger.info('Training role=%s mode=%s → %s', role, training_mode, adapter_out)
        log_path = adapters_base / f'{role}_{training_mode}_train.log'
        rc = run_llamafactory_train(cfg_path, log_file=log_path)
        if rc != 0:
            logger.error('Training failed for role=%s (rc=%d). Check log: %s', role, rc, log_path)
            continue

        adapter_paths[role] = adapter_out
        logger.info('Adapter saved: %s → %s', role, adapter_out)

    # Save adapter registry
    write_json(adapters_base / 'adapter_registry.json', adapter_paths)
    return adapter_paths


def train_dpo(
    config: dict[str, Any],
    export_dir: str,
    sft_adapter_path: str,
    dpo_output_dir: str,
) -> str | None:
    """Run DPO training on repair-derived preference pairs.

    Args:
        config: Full project config dict.
        export_dir: Directory containing preference_pairs.jsonl + dataset_info.json.
        sft_adapter_path: Path to the SFT adapter to continue from.
        dpo_output_dir: Output directory for DPO adapter.

    Returns:
        Path to the DPO adapter, or None if no preference data.
    """
    from cegsr.utils.modeling import resolve_local_model_path

    pref_file = Path(export_dir) / 'preference_pairs.jsonl'
    if not pref_file.exists() or pref_file.stat().st_size == 0:
        logger.info('No preference pairs found, skipping DPO')
        return None

    training_cfg = config.get('training', {})
    model_path = resolve_local_model_path(
        training_cfg.get('model_name_or_path', config['backend']['model_name_or_path']),
        model_size_hint=training_cfg.get('model_size', config['backend'].get('model_size')),
    )
    user_template = training_cfg.get('qlora_template', {})

    dpo_cfg = generate_dpo_config(
        model_name_or_path=model_path,
        adapter_name_or_path=sft_adapter_path,
        dataset_dir=str(Path(export_dir)),
        dataset_name='repair_preference',
        output_dir=dpo_output_dir,
        template=user_template.get('template', 'qwen'),
        quantization_bit=user_template.get('quantization_bit', 4),
        lora_rank=user_template.get('lora_rank', 16),
        lora_alpha=user_template.get('lora_alpha', 32),
    )
    cfg_path = Path(export_dir) / 'train_dpo.yaml'
    write_yaml(cfg_path, dpo_cfg)

    logger.info('Starting DPO training → %s', dpo_output_dir)
    log_path = Path(dpo_output_dir).parent / 'dpo_train.log'
    rc = run_llamafactory_train(cfg_path, log_file=log_path)
    if rc != 0:
        logger.error('DPO training failed (rc=%d). Check log: %s', rc, log_path)
        return None

    return dpo_output_dir
