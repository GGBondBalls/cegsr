#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "../.." && pwd)"
cd "$ROOT_DIR"

python scripts/run_ablation.py --config configs/profiles/dual_4090_vllm.yaml
