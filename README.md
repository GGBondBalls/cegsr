# CEG-SR: Causal Experience Graph with Selective Repair

CEG-SR is a from-scratch Python research framework for **self-evolving multi-agent systems** centered on three tightly coupled ideas:

- **Causal Experience Graph**
- **Fine-grained Credit Assignment**
- **Selective Repair**

This repository is **not** a patch over the original SiriuS implementation and does **not** copy its directory layout. It keeps the high-level spirit of bootstrapped multi-agent improvement, but rebuilds the method and engineering stack around a new paper-oriented direction:

> **CEG-SR = Causal Experience Graph with Selective Repair**

---

## 1. What this version focuses on

This version is intentionally scoped to **collaborative problem solving / question answering / reasoning** rather than competitive games or unrelated Sirius branches.

It provides:

- configurable multi-role agent graphs
- full episode / turn / subtrajectory trajectory logging
- turn-level, subtrajectory-level, and role-level credit assignment
- selective local repair instead of whole-trajectory rewrite
- causal experience graph building and graph-aware retrieval
- role-specific SFT export + preference/reward placeholders
- local training integration for **LLaMA-Factory**
- baseline and ablation execution
- report artifacts for paper writing

---

## 2. Relation to SiriuS

Conceptual overlap:
- multi-agent self-improvement loop
- trajectory collection
- improvement from prior runs

Key differences:
- **SiriuS-style whole successful trajectory retention** becomes **causal experience graph construction**
- **whole failed trajectory feedback rewrite** becomes **selective local repair**
- **trajectory-level supervision** becomes **turn / subtrajectory / role credit fusion**
- **OpenAI-oriented examples** are replaced by **local Transformers / vLLM / SGLang + LLaMA-Factory export**

---

## 3. Repository structure

```text
cegsr_project/
  configs/
    base.yaml                  # default real experiment config
    base_demo.yaml             # smoke-test config using mock backend
    datasets/
      reasoning_mix_eval.yaml
      reasoning_mix_train.yaml
    models/
      hf_local.yaml
      vllm.yaml
      sglang.yaml
  docs/
  scripts/
    prepare_data.py
    run_collect.py
    run_credit.py
    run_repair.py
    build_graph.py
    export_sft.py
    run_eval.py
    run_ablation.py
  src/cegsr/
    agents/
    backends/
    credit/
    data/
    evaluation/
    experience/
    repair/
    tasks/
    training/
    trajectories/
    workflows.py
  tests/
  examples/
  outputs/
```

---

## 4. Installation

### Minimal
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
pip install -e .
```

### With local model support
```bash
pip install -e ".[local,test]"
```

If you plan to build real benchmark mixtures from Hugging Face datasets, `datasets` is already included in the default requirements.

---

## 5. Real dataset preparation

The old `examples/sample_dataset.jsonl` is kept only for smoke tests. The default experiment config now expects a **real benchmark mixture** at:

```text
outputs/data/reasoning_mix_eval.jsonl
```

Build it with:

```bash
python scripts/prepare_data.py --config configs/datasets/reasoning_mix_eval.yaml
```

And build a larger train split with:

```bash
python scripts/prepare_data.py --config configs/datasets/reasoning_mix_train.yaml
```

### Included recipe: `reasoning_mix`

This builder unifies multiple public reasoning datasets into the common `TaskSample` schema:

- GSM8K
- CommonsenseQA
- ARC-Challenge
- BoolQ
- PubMedQA (best effort; skipped automatically if unavailable in your environment)

Each row includes metadata such as dataset name, source split, and category, so later reports can produce per-dataset breakdowns.

---

## 6. Default local inference path

The default config now assumes **local Transformers inference** with a Windows Hugging Face cache repo root like:

```text
/home/fyk/models/Qwen/Qwen2.5-X.XB-Instruct
```

Important details:
- you do **not** need to hardcode a `snapshots/<hash>` path yourself
- CEG-SR will automatically resolve the repo root to the latest valid snapshot directory
- `X.XB` is treated as a size placeholder and can be replaced through config, for example `7B`, `14B`, `32B`

The default `configs/base.yaml` uses:

```yaml
backend:
  kind: hf_local
  model_name_or_path: /home/fyk/models/Qwen/Qwen2.5-X.XB-Instruct
  model_size: 7B
```

If your actual local size is 14B, just change `model_size: 14B`.

---

## 7. Local inference backends

### 7.1 HF local direct inference
The `HFLocalBackend` now:
- resolves Hugging Face cache roots into `snapshots/*`
- prefers `tokenizer.apply_chat_template(...)`
- falls back to a deterministic mock backend if local loading fails

### 7.2 vLLM OpenAI-compatible server
Example:

```bash
vllm serve /path/to/model --dtype auto --api-key EMPTY
```

Then set:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=EMPTY
```

And use `configs/models/vllm.yaml`.

### 7.3 SGLang OpenAI-compatible server
Set:

```bash
export SGLANG_BASE_URL=http://localhost:30000/v1
export SGLANG_API_KEY=EMPTY
```

And use `configs/models/sglang.yaml`.

---

## 8. Core workflow

### Step 0: prepare real benchmark data
```bash
python scripts/prepare_data.py --config configs/datasets/reasoning_mix_eval.yaml
```

### Step 1: collect trajectories
```bash
python scripts/run_collect.py --config configs/base.yaml --output outputs/demo/raw.jsonl
```

### Step 2: annotate fine-grained credit
```bash
python scripts/run_credit.py --config configs/base.yaml --episodes outputs/demo/raw.jsonl --output outputs/demo/annotated.jsonl
```

### Step 3: selective repair
```bash
python scripts/run_repair.py --config configs/base.yaml --episodes outputs/demo/annotated.jsonl --output outputs/demo/repaired.jsonl
```

### Step 4: build causal experience graph
```bash
python scripts/build_graph.py --config configs/base.yaml --episodes outputs/demo/repaired.jsonl --graph-dir outputs/demo/graph
```

### Step 5: export role-specific training data
```bash
python scripts/export_sft.py --config configs/base.yaml --episodes outputs/demo/repaired.jsonl --export-dir outputs/demo/training_data
```

### Step 6: evaluate
```bash
python scripts/run_eval.py --episodes outputs/demo/repaired.jsonl --output-dir outputs/demo/eval --graph-dir outputs/demo/graph
```

### One-command pipeline
```bash
python cegsr/scripts/run_pipeline.py --config cegsr/configs/base.yaml
```

### Step 7: run ablations
```bash
python scripts/run_ablation.py --config configs/base.yaml --output-dir outputs/ablations_real
```

---

## 9. LLaMA-Factory training integration

After export, the project generates:
- `dataset_info.json`
- per-role `llamafactory_<role>_<mode>.yaml`
- `run_llamafactory.sh`

Exported formats include:
- role-specific SFT data
- repair-derived preference pairs
- reward / KTO-style placeholder data

Typical usage:

```bash
bash outputs/demo/training_data/run_llamafactory.sh
```

If your config includes `training.distributed`, the export step also writes:

```bash
bash cegsr/outputs/demo/training_data/run_llamafactory_ddp.sh
```

---

## 10. Baselines and ablations

Implemented baselines:
- `single_agent`
- `static_multi_agent`
- `sirius_lite`

Implemented ablations:
- `ours_wo_graph`
- `ours_wo_selective_repair`
- `trajectory_level_credit`
- `offline_sft_only`
- `repair_only`
- `ours_full`

---

## 11. Output artifacts

Each run can generate:
- `metrics.json`
- `metrics.csv`
- `report.md`
- `dataset_breakdown.csv`
- `error_cases.json`
- `error_cases.csv`
- ablation tables in csv/json/markdown

These are designed to be directly usable for paper-writing tables and case studies.

---

## 12. Quick start modes

### Smoke test only
```bash
python scripts/run_ablation.py --config configs/base_demo.yaml --output-dir outputs/ablations_demo
```

### Real local experiment
```bash
python scripts/prepare_data.py --config configs/datasets/reasoning_mix_eval.yaml
python scripts/run_ablation.py --config configs/base.yaml --output-dir outputs/ablations_real
```

### Dual-4090 server experiment
```bash
python cegsr/scripts/setup_experiment.py --config cegsr/configs/profiles/dual_4090_vllm.yaml
bash cegsr/outputs/dual_4090/launch_inference_server.sh
python cegsr/scripts/run_pipeline.py --config cegsr/configs/profiles/dual_4090_vllm.yaml
bash cegsr/outputs/dual_4090/training_data/run_llamafactory_ddp.sh
```

See `cegsr/docs/dual_4090_workflow.md` for the recommended repeated-experiment workflow.

---

## 13. Recommended next upgrades

The current version is intentionally a strong, runnable V1 rather than an overcomplicated research skeleton. The most valuable next steps are:

1. stronger verifier prompting / verifier calibration
2. learning-based subtrajectory segmentation
3. graph-aware iterative multi-round bootstrapping
4. checkpoint registry + per-role model routing after SFT
5. larger benchmark mixtures and per-domain evaluator hooks
