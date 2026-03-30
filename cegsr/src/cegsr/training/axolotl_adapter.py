from __future__ import annotations

from pathlib import Path
from typing import Any

from cegsr.utils.io import ensure_dir, write_yaml


def generate_axolotl_config(
    export_dir: str,
    dataset_file: str,
    model_name_or_path: str,
    output_path: str,
) -> str:
    """Optional Axolotl placeholder adapter."""
    export_dir_path = ensure_dir(export_dir)
    cfg: dict[str, Any] = {
        "base_model": model_name_or_path,
        "datasets": [{"path": str(Path(dataset_file).name), "type": "chat_template"}],
        "dataset_prepared_path": str(export_dir_path / "axolotl_cache"),
        "output_dir": output_path,
        "adapter": "lora",
        "sequence_len": 2048,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_epochs": 1,
    }
    cfg_path = export_dir_path / "axolotl_config.yaml"
    write_yaml(cfg_path, cfg)
    return str(cfg_path)
