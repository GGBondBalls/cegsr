#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0,1"
export FORCE_TORCHRUN=1
export NNODES="1"
export NODE_RANK="0"
export NPROC_PER_NODE="2"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"

llamafactory-cli train outputs/dual_4090_paper/training_data/llamafactory_planner_lora.yaml
llamafactory-cli train outputs/dual_4090_paper/training_data/llamafactory_planner_qlora.yaml
llamafactory-cli train outputs/dual_4090_paper/training_data/llamafactory_solver_lora.yaml
llamafactory-cli train outputs/dual_4090_paper/training_data/llamafactory_solver_qlora.yaml
llamafactory-cli train outputs/dual_4090_paper/training_data/llamafactory_verifier_lora.yaml
llamafactory-cli train outputs/dual_4090_paper/training_data/llamafactory_verifier_qlora.yaml
llamafactory-cli train outputs/dual_4090_paper/training_data/llamafactory_summarizer_lora.yaml
llamafactory-cli train outputs/dual_4090_paper/training_data/llamafactory_summarizer_qlora.yaml
