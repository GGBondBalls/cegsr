#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_data.py --config configs/datasets/reasoning_mix_eval.yaml
python scripts/run_collect.py --config configs/base.yaml --output outputs/demo/raw.jsonl
python scripts/run_credit.py --config configs/base.yaml --episodes outputs/demo/raw.jsonl --output outputs/demo/annotated.jsonl
python scripts/run_repair.py --config configs/base.yaml --episodes outputs/demo/annotated.jsonl --output outputs/demo/repaired.jsonl
python scripts/build_graph.py --config configs/base.yaml --episodes outputs/demo/repaired.jsonl --graph-dir outputs/demo/graph
python scripts/export_sft.py --config configs/base.yaml --episodes outputs/demo/repaired.jsonl --export-dir outputs/demo/training_data
python scripts/run_eval.py --episodes outputs/demo/repaired.jsonl --output-dir outputs/demo/eval --graph-dir outputs/demo/graph
python scripts/run_ablation.py --config configs/base.yaml --output-dir outputs/ablations_real
