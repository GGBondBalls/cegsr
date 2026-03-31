from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from cegsr.config.loader import load_config
from cegsr.utils.io import ensure_dir
from cegsr.utils.modeling import render_model_path_template


def _normalize_gpu_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _quote_command(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if part is not None and str(part) != "")


def _script_header(script_dir: Path, repo_root: Path) -> list[str]:
    root_target = os.path.relpath(repo_root, script_dir).replace("\\", "/")
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/{root_target}" && pwd)"',
        'cd "$ROOT_DIR"',
        "",
    ]


def _config_reference(config_path: Path, repo_root: Path) -> str:
    return os.path.relpath(config_path, repo_root).replace("\\", "/")


def _project_relative(path: Path, repo_root: Path) -> str:
    return os.path.relpath(path, repo_root).replace("\\", "/")


def _build_vllm_server_script(script_dir: Path, repo_root: Path, serving: dict[str, Any]) -> str:
    model_name_or_path = serving.get("model_name_or_path") or serving.get("model")
    if not model_name_or_path:
        raise ValueError("serving.model_name_or_path is required for vLLM launcher generation")

    gpu_list = _normalize_gpu_list(serving.get("gpu_ids"))
    tensor_parallel_size = int(serving.get("tensor_parallel_size", max(1, len(gpu_list)) or 1))
    command = [
        "vllm",
        "serve",
        render_model_path_template(model_name_or_path, serving.get("model_size")),
        "--host",
        serving.get("host", "127.0.0.1"),
        "--port",
        serving.get("port", 8000),
        "--dtype",
        serving.get("dtype", "auto"),
        "--tensor-parallel-size",
        tensor_parallel_size,
        "--gpu-memory-utilization",
        serving.get("gpu_memory_utilization", 0.9),
        "--api-key",
        serving.get("api_key", "EMPTY"),
    ]
    optional_flags = {
        "--max-model-len": serving.get("max_model_len"),
        "--swap-space": serving.get("swap_space"),
        "--max-num-seqs": serving.get("max_num_seqs"),
    }
    for flag, value in optional_flags.items():
        if value is not None:
            command.extend([flag, value])
    for item in serving.get("extra_args", []):
        command.append(item)

    lines = _script_header(script_dir, repo_root)
    if gpu_list:
        lines.append(f'export CUDA_VISIBLE_DEVICES="{",".join(gpu_list)}"')
        lines.append("")
    lines.append(_quote_command(command))
    return "\n".join(lines) + "\n"


def generate_experiment_scripts(config_path: str | Path, output_dir: str | None = None) -> dict[str, str]:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    repo_root = Path.cwd().resolve()
    script_dir = ensure_dir(output_dir or config["project"]["output_dir"]).resolve()
    config_ref = _config_reference(config_file, repo_root)

    scripts: dict[str, str] = {}

    pipeline_lines = _script_header(script_dir, repo_root)
    pipeline_lines.append(_quote_command(["python", "scripts/run_pipeline.py", "--config", config_ref]))
    pipeline_path = script_dir / "run_pipeline.sh"
    pipeline_path.write_text("\n".join(pipeline_lines) + "\n", encoding="utf-8")
    scripts["pipeline"] = _project_relative(pipeline_path, repo_root)

    ablation_lines = _script_header(script_dir, repo_root)
    ablation_lines.append(_quote_command(["python", "scripts/run_ablation.py", "--config", config_ref]))
    ablation_path = script_dir / "run_ablation.sh"
    ablation_path.write_text("\n".join(ablation_lines) + "\n", encoding="utf-8")
    scripts["ablation"] = _project_relative(ablation_path, repo_root)

    serving = config.get("serving", {})
    if serving.get("enabled") and serving.get("kind", "vllm") == "vllm":
        launcher_path = script_dir / "launch_inference_server.sh"
        launcher_path.write_text(_build_vllm_server_script(script_dir, repo_root, serving), encoding="utf-8")
        scripts["serving"] = _project_relative(launcher_path, repo_root)

    return scripts
