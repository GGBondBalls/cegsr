# Ablation Table

| method | accuracy | exact_match | repair_coverage | retrieval_proxy |
|---|---:|---:|---:|---:|
| single_agent | 0.2943 | 0.0457 | 0.0 | 0.0 |
| static_multi_agent | 0.7243 | 0.2229 | 0.0 | 0.0 |
| sirius_lite | 0.9543 | 0.2986 | 0.0 | 0.0 |
| ours_wo_graph | 0.8043 | 0.2429 | 0.2814 | 0.0 |
| ours_wo_selective_repair | 0.7171 | 0.2086 | 0.0 | 0.7198 |
| trajectory_level_credit | 0.7229 | 0.2271 | 0.0 | 0.0 |
| repair_only | 0.8043 | 0.2429 | 0.2814 | 0.0 |
| offline_sft_only | 0.73 | 0.2314 | 0.0 | 0.0 |
| ours_full | 0.72 | 0.2086 | 0.0 | 0.7228 |
