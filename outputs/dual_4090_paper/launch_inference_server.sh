#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="0,1"

if command -v vllm >/dev/null 2>&1; then
  vllm serve /home/fyk/models/Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --dtype auto --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --api-key EMPTY --max-model-len 4096 --max-num-seqs 16
elif python -c "import vllm" >/dev/null 2>&1; then
  python -m vllm.entrypoints.openai.api_server --model /home/fyk/models/Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --dtype auto --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --api-key EMPTY --max-model-len 4096 --max-num-seqs 16
else
  echo "vLLM is not available in the current environment." >&2
  echo "Install it first, for example: pip install vllm" >&2
  exit 127
fi
