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
    try:
        root_target = os.path.relpath(repo_root, script_dir).replace("\\", "/")
        root_expr = f'$(dirname "${{BASH_SOURCE[0]}}")/{root_target}'
    except ValueError:
        root_expr = str(repo_root).replace("\\", "/")
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'ROOT_DIR="$(cd "{root_expr}" && pwd)"',
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
    model_ref = render_model_path_template(model_name_or_path, serving.get("model_size"))
    serve_command = [
        "vllm",
        "serve",
        model_ref,
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
    module_command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_ref,
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
        "--max-num-seqs": serving.get("max_num_seqs"),
    }
    for flag, value in optional_flags.items():
        if value is not None:
            serve_command.extend([flag, value])
            module_command.extend([flag, value])
    for item in serving.get("extra_args", []):
        serve_command.append(item)
        module_command.append(item)

    lines = _script_header(script_dir, repo_root)
    if gpu_list:
        lines.append(f'export CUDA_VISIBLE_DEVICES="{",".join(gpu_list)}"')
        lines.append("")
    lines.extend(
        [
            'if command -v vllm >/dev/null 2>&1; then',
            f"  {_quote_command(serve_command)}",
            'elif python -c "import vllm" >/dev/null 2>&1; then',
            f"  {_quote_command(module_command)}",
            "else",
            '  echo "vLLM is not available in the current environment." >&2',
            '  echo "Install it first, for example: pip install vllm" >&2',
            "  exit 127",
            "fi",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_prepare_data_script(script_dir: Path, repo_root: Path, prepare_config: str | None) -> str:
    lines = _script_header(script_dir, repo_root)
    if prepare_config:
        lines.append(_quote_command(["python", "scripts/prepare_data.py", "--config", prepare_config]))
    else:
        lines.extend(
            [
                'echo "No task.prepare_config is defined in the current config." >&2',
                "exit 1",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_inference_healthcheck_lines(config: dict[str, Any]) -> list[str]:
    backend = config.get("backend", {})
    kind = backend.get("kind")
    if kind not in {"vllm", "sglang"}:
        return []

    base_url = str(backend.get("base_url", "")).rstrip("/")
    if not base_url:
        return []

    models_url = f"{base_url}/models"
    api_key = str(backend.get("api_key", "") or "")
    launch_hint = Path(config.get("project", {}).get("output_dir", "outputs/dual_4090")) / "launch_inference_server.sh"
    return [
        'python - <<\'PY\'',
        "import sys",
        "import requests",
        f"url = {models_url!r}",
        f"api_key = {api_key!r}",
        f"launch_hint = {str(launch_hint).replace(chr(92), '/').__repr__()}",
        "headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}",
        "try:",
        "    response = requests.get(url, headers=headers, timeout=5)",
        "    response.raise_for_status()",
        "except Exception as exc:",
        '    print(f\"Inference server is not reachable: {url}\", file=sys.stderr)',
        '    print(f\"Reason: {exc}\", file=sys.stderr)',
        '    print(f\"Start it first with: bash {launch_hint}\", file=sys.stderr)',
        "    raise SystemExit(1)",
        "PY",
        "",
    ]


def generate_experiment_scripts(config_path: str | Path, output_dir: str | None = None) -> dict[str, str]:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    repo_root = Path.cwd().resolve()
    script_dir = ensure_dir(output_dir or config["project"]["output_dir"]).resolve()
    config_ref = _config_reference(config_file, repo_root)

    scripts: dict[str, str] = {}

    prepare_path = script_dir / "prepare_data.sh"
    prepare_path.write_text(
        _build_prepare_data_script(script_dir, repo_root, config.get("task", {}).get("prepare_config")),
        encoding="utf-8",
    )
    scripts["prepare_data"] = _project_relative(prepare_path, repo_root)

    pipeline_lines = _script_header(script_dir, repo_root)
    dataset_path = config.get("task", {}).get("dataset_path")
    if dataset_path:
        pipeline_lines.extend(
            [
                f'if [ ! -f "{dataset_path}" ]; then',
                f'  echo "Dataset not found: {dataset_path}" >&2',
                f'  echo "Run: bash {prepare_path.name}" >&2',
                "  exit 1",
                "fi",
                "",
            ]
        )
    pipeline_lines.extend(_build_inference_healthcheck_lines(config))
    pipeline_lines.append(_quote_command(["python", "scripts/run_pipeline.py", "--config", config_ref]))
    pipeline_path = script_dir / "run_pipeline.sh"
    pipeline_path.write_text("\n".join(pipeline_lines) + "\n", encoding="utf-8")
    scripts["pipeline"] = _project_relative(pipeline_path, repo_root)

    ablation_lines = _script_header(script_dir, repo_root)
    ablation_lines.extend(_build_inference_healthcheck_lines(config))
    ablation_lines.append(_quote_command(["python", "scripts/run_ablation.py", "--config", config_ref]))
    ablation_path = script_dir / "run_ablation.sh"
    ablation_path.write_text("\n".join(ablation_lines) + "\n", encoding="utf-8")
    scripts["ablation"] = _project_relative(ablation_path, repo_root)

    serving = config.get("serving", {})
    if serving.get("enabled") and serving.get("kind", "vllm") == "vllm":
        launcher_path = script_dir / "launch_inference_server.sh"
        launcher_path.write_text(_build_vllm_server_script(script_dir, repo_root, serving), encoding="utf-8")
        scripts["serving"] = _project_relative(launcher_path, repo_root)

    # Iterative training script
    iterative_lines = _script_header(script_dir, repo_root)
    iterative_lines.extend(_build_inference_healthcheck_lines(config))
    iterative_lines.append(
        _quote_command([
            "python", "scripts/run_iterative.py",
            "--config", config_ref,
            "--max-iterations", "3",
            "--mode", config.get("training", {}).get("mode", "qlora"),
        ])
    )
    iterative_path = script_dir / "run_iterative.sh"
    iterative_path.write_text("\n".join(iterative_lines) + "\n", encoding="utf-8")
    scripts["iterative"] = _project_relative(iterative_path, repo_root)

    return scripts