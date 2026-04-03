#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python scripts/check_model_server.py --base-url http://127.0.0.1:8000/v1 --api-key EMPTY
