# Experiment Guide

## Goal

This guide describes the recommended first real experiment path for CEG-SR after the smoke-test stage.

Run the commands below from the repository root so `scripts/...`, `configs/...`, and `outputs/...` stay relative. Model paths remain absolute, for example `/home/fyk/models/Qwen/Qwen2.5-7B-Instruct`.

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

If you want a single entrypoint for the same sequence, use:

```bash
python scripts/run_pipeline.py --config configs/base.yaml
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

If your config defines `training.distributed`, the export step also generates:

```bash
bash outputs/demo/training_data/run_llamafactory_ddp.sh
```

## 7. Dual-4090 server workflow

For a dual-RTX4090 server, prefer the dedicated profile:

```bash
python scripts/setup_experiment.py --config configs/profiles/dual_4090_vllm.yaml
bash outputs/dual_4090/prepare_data.sh
bash outputs/dual_4090/launch_inference_server.sh
bash outputs/dual_4090/run_pipeline.sh
bash outputs/dual_4090/training_data/run_llamafactory_ddp.sh
```

See [`dual_4090_workflow.md`](dual_4090_workflow.md) for details.

## 8. Practical recommendation

First run with `model_size: 7B` to validate the loop, then scale to a stronger local checkpoint. Keep the benchmark mix fixed while iterating on verifier prompts, repair thresholds, and graph retrieval settings.
