# Architecture

CEG-SR is organized as a lightweight Python package with explicit boundaries:

- `backends/`: unified local inference interfaces for HF local, vLLM, and SGLang.
- `agents/`: role agents and the configurable agent graph runtime.
- `tasks/`: prompt construction and evaluation logic.
- `trajectories/`: schemas, replay, recording, and rule-based segmentation.
- `credit/`: modular fine-grained credit signals and weighted fusion.
- `repair/`: detector + selective local patching.
- `experience/`: causal experience graph building, storage, and retrieval.
- `training/`: SFT / preference / reward export and LLaMA-Factory integration.
- `evaluation/`: metrics and report generation.
- `baselines/`: single-agent, static multi-agent, Sirius-lite.

The runtime is intentionally simple:
1. run a configurable agent graph,
2. record a full episode trajectory,
3. assign multi-source credit,
4. repair only low-credit local spans,
5. store high-value turns/subtrajectories in a causal experience graph,
6. re-inject retrieved experiences into later runs,
7. export training data for role-specific local SFT.
