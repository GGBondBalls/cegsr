# Experiment Guide

## Goal

This guide describes the recommended first real experiment path for CEG-SR after the smoke-test stage.

## 1. Build the benchmark mixture

```bash
python scripts/prepare_data.py --config configs/datasets/reasoning_mix_eval.yaml
python scripts/prepare_data.py --config configs/datasets/reasoning_mix_train.yaml
```

The evaluation mix unifies GSM8K, CommonsenseQA, ARC-Challenge, BoolQ, and PubMedQA into the common `TaskSample` format.

## 2. Configure the local model

The default config uses a Windows-local HF cache root:

```yaml
backend:
  kind: hf_local
  model_name_or_path: /home/fyk/models/Qwen/Qwen2.5-X.XB-Instruct
  model_size: 7B
```

If your local model is 14B, set `model_size: 14B`. The backend will resolve the repo root into the newest valid `snapshots/*` directory automatically.

## 3. Run the collect → annotate → repair → graph → export → eval pipeline

```bash
python scripts/run_collect.py --config configs/base.yaml --output outputs/demo/raw.jsonl
python scripts/run_credit.py --config configs/base.yaml --episodes outputs/demo/raw.jsonl --output outputs/demo/annotated.jsonl
python scripts/run_repair.py --config configs/base.yaml --episodes outputs/demo/annotated.jsonl --output outputs/demo/repaired.jsonl
python scripts/build_graph.py --config configs/base.yaml --episodes outputs/demo/repaired.jsonl --graph-dir outputs/demo/graph
python scripts/export_sft.py --config configs/base.yaml --episodes outputs/demo/repaired.jsonl --export-dir outputs/demo/training_data
python scripts/run_eval.py --episodes outputs/demo/repaired.jsonl --output-dir outputs/demo/eval --graph-dir outputs/demo/graph
```

## 4. Run the baseline / ablation suite

```bash
python scripts/run_ablation.py --config configs/base.yaml --output-dir outputs/ablations_real
```

## 5. Inspect artifacts

Important outputs:
- `outputs/demo/eval/metrics.json`
- `outputs/demo/eval/dataset_breakdown.csv`
- `outputs/demo/eval/error_cases.csv`
- `outputs/ablations_real/ablation_table.csv`

## 6. Training handoff to LLaMA-Factory

The export step generates `dataset_info.json` and per-role training YAMLs. Run:

```bash
bash outputs/demo/training_data/run_llamafactory.sh
```

## 7. Practical recommendation

First run with `model_size: 7B` to validate the loop, then scale to a stronger local checkpoint. Keep the benchmark mix fixed while iterating on verifier prompts, repair thresholds, and graph retrieval settings.
