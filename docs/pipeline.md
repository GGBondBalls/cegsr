# Pipeline

## Overview

CEG-SR is built around the following workflow:

1. **collect**: run the multi-agent graph and record raw trajectories
2. **annotate**: segment trajectories and compute fine-grained credit
3. **repair**: selectively repair low-credit problematic spans
4. **build_memory**: construct the causal experience graph
5. **export**: export role-specific SFT / preference / reward-style data
6. **train**: hand off to LLaMA-Factory
7. **eval**: evaluate a chosen episode file and export reports
8. **ablate**: compare baselines and ablations automatically

## Why the order matters

The causal experience graph is built from trajectories that already carry fused credit and selective repair records. This makes the graph semantically different from simply storing all successful trajectories: nodes are filtered and typed by local value.

## Current V1 design choice

This version emphasizes a fully runnable and inspectable pipeline over hard-to-verify sophistication. That means:
- segmentation is rule-based
- credit signals are heuristic but modular
- graph retrieval is embedding + metadata filter + neighborhood expansion
- repair targets the earliest low-credit segment and replays only the dependent suffix
