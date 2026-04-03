# Ablation Table

| method | accuracy | exact_match | repair_coverage | retrieval_proxy |
|---|---:|---:|---:|---:|
| single_agent | 0.42 | 0.105 | 0.0 | 0.0 |
| static_multi_agent | 0.6825 | 0.3175 | 0.0 | 0.0 |
| sirius_lite | 0.7875 | 0.36 | 0.0 | 0.0 |
| ours_wo_graph | 0.7025 | 0.3275 | 0.3425 | 0.0 |
| ours_wo_selective_repair | 0.68 | 0.34 | 0.0 | 0.6896 |
| trajectory_level_credit | 0.6775 | 0.3175 | 0.0 | 0.0 |
| repair_only | 0.7025 | 0.3275 | 0.3425 | 0.0 |
| offline_sft_only | 0.6775 | 0.3175 | 0.0 | 0.0 |
| ours_full | 0.68 | 0.34 | 0.0 | 0.6896 |
