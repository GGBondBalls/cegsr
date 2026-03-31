# Dual-4090 Workflow

## Goal

This note captures the recommended server-side workflow for a dual-RTX4090 machine:

- use a persistent vLLM server for inference-heavy stages
- keep the CEG-SR pipeline as a single command
- export LLaMA-Factory configs that already include a 2-GPU DDP launcher

This keeps the project structure unchanged while making repeated experiments easier to run.

All commands below assume you run them from the repository root `~/cegsr`, so script and config paths stay relative. The only paths intentionally kept absolute are model directories such as `/home/fyk/models/Qwen/Qwen2.5-7B-Instruct`.

## Recommended profile

Use:

```text
configs/profiles/dual_4090_vllm.yaml
```

This profile does three things:

1. switches the online inference backend to `vllm`
2. reserves both GPUs for the inference server with `tensor_parallel_size: 2`
3. prepares 14B training export with a generated `run_llamafactory_ddp.sh`

Inference and training can still target different sizes:

- `backend.model_size: 7B`
- `training.model_size: 14B`

## Why this layout

For repeated `collect / repair / ablation` experiments, loading a Hugging Face model inside every Python script wastes time and GPU memory churn. A persistent vLLM server avoids repeated model loads and is a better default once you move beyond tiny smoke tests.

For training, the project now exports:

- `run_llamafactory.sh`
- `run_llamafactory_ddp.sh`

The second script sets `CUDA_VISIBLE_DEVICES`, `FORCE_TORCHRUN`, `NPROC_PER_NODE`, `MASTER_ADDR`, and `MASTER_PORT` for a single-node 2-GPU run.

## One-time script generation

```bash
python scripts/setup_experiment.py --config configs/profiles/dual_4090_vllm.yaml
```

This writes helper scripts into:

```text
outputs/dual_4090/
```

Generated files:

- `launch_inference_server.sh`
- `run_pipeline.sh`
- `run_ablation.sh`

## Recommended execution order

Start the inference server first:

```bash
bash outputs/dual_4090/launch_inference_server.sh
```

Then run the full experiment pipeline:

```bash
bash outputs/dual_4090/run_pipeline.sh
```

Or run directly without the wrapper:

```bash
python scripts/run_pipeline.py --config configs/profiles/dual_4090_vllm.yaml
```

After export completes, train with:

```bash
bash outputs/dual_4090/training_data/run_llamafactory_ddp.sh
```

## Common adjustments

### Change the inference model

Edit only:

```yaml
backend:
  model_size: 7B

serving:
  model_size: 7B
```

### Change the training model

Edit only:

```yaml
training:
  model_size: 14B
```

If 14B full LoRA is too heavy, keep using the provided QLoRA template.

### Keep outputs separated across runs

Set:

```yaml
project:
  output_dir: outputs/<new_run_name>
```

Then rerun `setup_experiment.py` for the new profile.
