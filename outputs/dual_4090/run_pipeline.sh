#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f "outputs/data/reasoning_mix_eval.jsonl" ]; then
  echo "Dataset not found: outputs/data/reasoning_mix_eval.jsonl" >&2
  echo "Run: bash prepare_data.sh" >&2
  exit 1
fi

python - <<'PY'
import sys
import requests
url = 'http://127.0.0.1:8000/v1/models'
api_key = 'EMPTY'
headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
try:
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()
except Exception as exc:
    print(f"Inference server is not reachable: {url}", file=sys.stderr)
    print(f"Reason: {exc}", file=sys.stderr)
    print("Start it first with: bash outputs/dual_4090/launch_inference_server.sh", file=sys.stderr)
    raise SystemExit(1)
PY

python scripts/run_pipeline.py --config configs/profiles/dual_4090_vllm.yaml
