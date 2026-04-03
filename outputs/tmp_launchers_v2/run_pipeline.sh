#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f "outputs/data/reasoning_mix_eval.jsonl" ]; then
  echo "Dataset not found: outputs/data/reasoning_mix_eval.jsonl" >&2
  echo "Run: bash prepare_data.sh" >&2
  exit 1
fi

python scripts/run_pipeline.py --config configs/profiles/dual_4090_vllm.yaml
