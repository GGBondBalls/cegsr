# Ablation Table

| method | accuracy | exact_match | repair_coverage | retrieval_proxy |
|---|---:|---:|---:|---:|
| single_agent | 0.4525 | 0.105 | 0.0 | 0.0 |
| static_multi_agent | 0.6925 | 0.325 | 0.0 | 0.0 |
| sirius_lite | 0.7875 | 0.365 | 0.0 | 0.0 |
| ours_wo_graph | 0.6975 | 0.3325 | 0.0525 | 0.0 |
| ours_wo_selective_repair | 0.64 | 0.3175 | 0.0 | 0.64 |
| trajectory_level_credit | 0.695 | 0.3325 | 0.0 | 0.0 |
| repair_only | 0.6975 | 0.3325 | 0.0525 | 0.0 |
| offline_sft_only | 0.695 | 0.3325 | 0.0 | 0.0 |
| ours_full | 0.6375 | 0.3175 | 0.0 | 0.6375 |
