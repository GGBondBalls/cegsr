#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "../.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="0,1"

vllm serve /home/fyk/models/Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --dtype auto --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --api-key EMPTY --max-model-len 4096 --swap-space 8 --max-num-seqs 16
