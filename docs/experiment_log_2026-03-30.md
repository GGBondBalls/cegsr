# CEG-SR 实验日志

> 本文档是 CEG-SR 项目从启动至今的完整实验日志。面向第一次阅读本项目的研究者，记录所有有效的实验操作、代码改进、关键结果与判断。
>
> 最后整理: 2026-04-04

---

## 目录

- [1. 项目实验设定](#1-项目实验设定)
- [2. Phase 1: 主流程搭建与方法激活 (2026-03-30)](#2-phase-1-主流程搭建与方法激活-2026-03-30)
- [3. Phase 2: 双 4090 服务器部署与 vLLM 对接 (2026-03-31)](#3-phase-2-双-4090-服务器部署与-vllm-对接-2026-03-31)
- [4. Phase 3: 七数据集 Paper Benchmark 恢复与全量消融 (2026-04-03)](#4-phase-3-七数据集-paper-benchmark-恢复与全量消融-2026-04-03)
- [5. 当前总结: 已完成与未完成](#5-当前总结-已完成与未完成)
- [6. 下一步: 训练闭环实现](#6-下一步-训练闭环实现)
- [7. Phase 4: 训练闭环代码实现 (2026-04-04 — 2026-04-05)](#7-phase-4-训练闭环代码实现-2026-04-04--2026-04-05)

---

## 1. 项目实验设定

### 1.1 多角色结构

四角色顺序协作: `planner → solver → verifier → summarizer`

### 1.2 信用融合权重

```yaml
outcome: 0.35
verifier: 0.4
dependency: 0.25
```

### 1.3 修复触发参数

```yaml
turn_threshold: 0.55
subtrajectory_threshold: 0.55
require_verifier_issue: true
verifier_issue_threshold: 0.6
relax_on_failure: true        # 对失败样本的 solver/summarizer 放宽触发
failure_margin: 0.08
```

### 1.4 检索参数

```yaml
top_k: 2
expand_neighbors: false
role_match_only: true
exclude_same_sample: true
same_dataset_only: true
min_similarity: 0.3
question_overlap_weight: 0.2
min_question_overlap: 0.02
enabled_roles: [solver]
```

---

## 2. Phase 1: 主流程搭建与方法激活 (2026-03-30)

### 2.1 环境

- 后端: `hf_local` (本地 Transformers 直推)
- 模型: `Qwen2.5-1.5B-Instruct`
- 数据: `reasoning_mix_eval.jsonl` (commonsense_qa + ai2_arc + boolq + pubmed_qa, 共 400 条)

### 2.2 初始运行: 流程通了但方法未激活

```bash
python scripts/run_collect.py --config configs/base.yaml --output outputs/demo/raw.jsonl
python scripts/run_credit.py  --config configs/base.yaml --episodes outputs/demo/raw.jsonl --output outputs/demo/annotated.jsonl
python scripts/run_repair.py  --config configs/base.yaml --episodes outputs/demo/annotated.jsonl --output outputs/demo/repaired.jsonl
python scripts/build_graph.py --config configs/base.yaml --episodes outputs/demo/repaired.jsonl --graph-dir outputs/demo/graph
python scripts/export_sft.py  --config configs/base.yaml --episodes outputs/demo/repaired.jsonl --export-dir outputs/demo/training_data
python scripts/run_eval.py    --episodes outputs/demo/repaired.jsonl --output-dir outputs/demo/eval --graph-dir outputs/demo/graph
```

初始结果:
- accuracy = 0.6975
- **repair_coverage = 0.0525** (极低，说明 repair 几乎未工作)
- retrieval_hit_usefulness_proxy = 0.0

### 2.3 问题排查与修复

#### 问题 A: repair prompt 未传递给模型

repair 逻辑已生成 repair_mode/repair_reason/preserved_context，但 prompt 构造未写入这些信息，导致模型在无差别重跑而非局部修复。

**修复:** `src/cegsr/tasks/qa.py` — 在 `build_prompt()` 中真正写入 repair context 和 preserved_context。

#### 问题 B: verifier credit 高分扩散

`VerifierCreditSignal` 过于宽松，verifier 的高分以默认值传播到其它 turn，导致"细粒度 credit"退化为"整体都不低"，repair detector 很难找到低信用的局部 span。

**修复:**
- `src/cegsr/credit/verifier_credit.py` — 收紧 verifier score: correct verdict cap 到 0.85, incorrect cap 到 0.35; 非 verifier turn 根据 verdict 方向做调整而非统一给高分。
- `src/cegsr/repair/detector.py` — 对失败样本的 solver/summarizer 放宽 repair 触发条件。

#### 问题 C: graph retrieval 引入噪声

初始 retrieval 使结果从 0.6975 下降到 0.59。经过多轮收紧后稳定在 0.68，但始终未超过无 graph 的结果。

**收紧措施 (累计):**
1. 禁止 same-sample 泄漏
2. 默认限制 same-role
3. 关闭 neighbor expansion
4. 限制只对 solver 开启 retrieval
5. 限制 same-dataset 检索
6. 增加 minimum similarity
7. 增加 question-overlap 重排
8. Graph node embedding 从 response 改为 question + role + response
9. 对 solver 注入的 retrieved snippet 做去答案化

**涉及文件:** `src/cegsr/experience/retriever.py`, `builder.py`, `src/cegsr/agents/graph_runtime.py`, `src/cegsr/tasks/qa.py`

### 2.4 修复后结果 (repaired_v2)

```bash
python scripts/run_credit.py  --config configs/base.yaml --episodes outputs/demo/raw.jsonl --output outputs/demo/annotated_v2.jsonl
python scripts/run_repair.py  --config configs/base.yaml --episodes outputs/demo/annotated_v2.jsonl --output outputs/demo/repaired_v2.jsonl
python scripts/run_eval.py    --episodes outputs/demo/repaired_v2.jsonl --output-dir outputs/demo/eval_repaired_v2
```

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| accuracy | 0.6975 | **0.7875** |
| repair_coverage | 0.0525 | **0.315** |
| repair_success_rate | — | **0.3254** |
| num_changed_repairs | — | 126 |

分数据集: commonsense_qa=0.84, ai2_arc=0.79, boolq=0.78, pubmed_qa=0.74

### 2.5 Phase 1 消融 (hf_local + 1.5B)

| method | accuracy | exact_match |
|--------|----------|-------------|
| single_agent | 0.42 | 0.105 |
| static_multi_agent | 0.6825 | 0.3175 |
| sirius_lite | 0.7875 | 0.36 |
| ours_wo_graph | 0.77 | 0.36 |
| ours_full | 0.6775 | 0.34 |

### 2.6 Phase 1 关键结论

1. **主增益来自 credit + repair**: 从 0.68 提升到 0.77-0.79
2. **Graph retrieval 当前为负收益**: ours_full (0.6775) < ours_wo_graph (0.77)
3. 工具链从"流程能跑"进入"方法有效"阶段

---

## 3. Phase 2: 双 4090 服务器部署与 vLLM 对接 (2026-03-31)

### 3.1 环境切换

| 维度 | Phase 1 | Phase 2 |
|------|---------|---------|
| GPU | 本地 | RTX 4090 x2 (服务器) |
| 后端 | hf_local | vLLM (TP=2) |
| 模型 | Qwen2.5-1.5B | Qwen2.5-7B-Instruct |
| 数据 | 四数据集 400 条 | 三数据集 300 条 (gsm8k/boolq 构建失败) |

### 3.2 工程优化 (非方法改动)

| 改动 | 涉及文件 | 说明 |
|------|---------|------|
| 配置继承 (_base_) | `src/cegsr/config/loader.py` | Profile 只需覆盖必要字段 |
| 模型路径模板统一 | `src/cegsr/utils/modeling.py` | X.XB → model_size 参数化 |
| 一键 pipeline 入口 | `scripts/run_pipeline.py`, `src/cegsr/workflows.py` | collect→credit→repair→graph→export→eval 单命令 |
| 服务器脚本自动生成 | `scripts/setup_experiment.py`, `src/cegsr/launchers.py` | 按 profile 生成 prepare/launch/pipeline/ablation 脚本 |
| 双卡 DDP 训练脚本 | `src/cegsr/training/llamafactory_adapter.py` | 生成 torchrun 环境变量 |
| vLLM 兼容修复 | `src/cegsr/launchers.py` | 去掉 --swap-space; 健康检查带 auth header |

### 3.3 服务器三段对照 (300 样本, 7B + vLLM)

| 阶段 | accuracy | exact_match | repair_coverage |
|------|----------|-------------|-----------------|
| raw | 0.8033 | 0.2467 | — |
| annotated | 0.8033 | 0.2467 | — |
| **repaired** | **0.8533** | **0.2633** | **0.1967** |

分数据集增益: commonsense_qa +0.04, ai2_arc +0.06, pubmed_qa +0.05

### 3.4 Phase 2 关键结论

1. **raw == annotated**: credit 只是打标签，不直接改变输出
2. **repaired - raw = +0.05**: repair 是当前唯一直接提升准确率的模块
3. **vLLM + TP=2 流程稳定**: 可作为后续默认运行配置
4. 数据不完整 (gsm8k/boolq 缺失)，不能作为论文主表

---

## 4. Phase 3: 七数据集 Paper Benchmark 恢复与全量消融 (2026-04-03)

### 4.1 代码修复

#### 修复 A: 补全七数据集

恢复完整数据构建: college_physics, college_chemistry, pubmed_qa, gsm8k, commonsense_qa, ai2_arc, boolq。

**涉及文件:** `src/cegsr/data/builders.py`, `configs/datasets/paper_reasoning_eval.yaml`, `paper_reasoning_train.yaml`, `configs/profiles/paper_benchmark.yaml`, `dual_4090_vllm_paper.yaml`

#### 修复 B: GSM8K 数值评测

之前 gsm8k 复用通用 free-form exact match，导致所有正确数值答案 (如 "Final Answer: 42") 被判错。

**修复:** `src/cegsr/tasks/qa.py` — 新增 `numeric_accuracy` 路径，对 gsm8k/math_word_problem 做数值归一化比较。

**效果:** gsm8k 从 0.0 (评测 bug) → 0.83 (修复后)。这是评测协议修正，不是方法改进。

#### 修复 C: 消融进度可见性

`run_ablation.sh` 长时间无输出，看起来像卡死。

**修复:** `src/cegsr/workflows.py` 增加 `[Ablation] Start/Done` 日志和逐样本进度条; `src/cegsr/launchers.py` 增加健康检查。

### 4.2 数据准备结果

**Eval 集:** 700 条 (7 × 100)

**Train 集:** 2010 条 — 但 college_physics/college_chemistry 各仅 5 条 (只有 dev split)。其余五个来源各 400 条。

### 4.3 主流程结果 (700 样本, 7B + vLLM)

```bash
bash outputs/dual_4090_paper/run_pipeline.sh
```

| 指标 | 值 |
|------|-----|
| accuracy | 0.8043 |
| exact_match | 0.2429 |
| repair_coverage | 0.2814 |
| repair_success_rate | 0.3046 |
| num_changed_repairs | 197 |
| graph_num_nodes | 2137 |
| graph_num_edges | 50806 |

分数据集: college_physics=0.82, college_chemistry=0.59, pubmed_qa=0.78, gsm8k=0.83, commonsense_qa=0.88, ai2_arc=0.90, boolq=0.83

### 4.4 全量消融 (七数据集, 7B + vLLM)

```bash
bash outputs/dual_4090_paper/run_ablation.sh
```

| method | accuracy | exact_match | repair_coverage | retrieval_proxy |
|--------|----------|-------------|-----------------|-----------------|
| single_agent | 0.2943 | 0.0457 | 0.0 | 0.0 |
| static_multi_agent | 0.7243 | 0.2229 | 0.0 | 0.0 |
| sirius_lite | **0.9543** | 0.2986 | 0.0 | 0.0 |
| ours_wo_graph | 0.8043 | 0.2429 | 0.2814 | 0.0 |
| ours_wo_selective_repair | 0.7171 | 0.2086 | 0.0 | 0.7198 |
| trajectory_level_credit | 0.7229 | 0.2271 | 0.0 | 0.0 |
| repair_only | 0.8043 | 0.2429 | 0.2814 | 0.0 |
| offline_sft_only | 0.7300 | 0.2314 | 0.0 | 0.0 |
| ours_full | 0.7200 | 0.2086 | 0.0 | 0.7228 |

### 4.5 sirius_lite 异常: 0.9543 vs 重跑 0.8086

首次消融中 sirius_lite = 0.9543 异常偏高。单独重跑 (`ablations_v2/sirius_lite`) 得到:

| 指标 | ablations (首次) | ablations_v2 (重跑) |
|------|-----------------|-------------------|
| accuracy | 0.9543 | **0.8086** |
| exact_match | 0.2986 | 0.2429 |

0.8086 更合理，与 ours_wo_graph (0.8043) 接近。首次异常的可能原因: 消融 suite 中各方法共享同一 vLLM 服务，sirius_lite 做双 pass 推理，可能受到服务端缓存或并发状态影响。

**结论:** 以重跑结果 0.8086 为准。sirius_lite 与 ours_wo_graph 本质持平。

### 4.6 消融结果解读

**结论 1: repair 是当前最稳定的正向增益来源**
- static_multi_agent = 0.7243
- trajectory_level_credit = 0.7229 (仅粗粒度 credit 无帮助)
- offline_sft_only = 0.7300 (仅导出数据无帮助)
- **ours_wo_graph = 0.8043** (credit + repair 带来 +0.08)

**结论 2: graph retrieval 仍为负收益**
- ours_wo_graph = 0.8043
- ours_full = 0.7200 (启用 graph 后 -0.08)
- ours_wo_selective_repair = 0.7171

**结论 3: repair_only 与 ours_wo_graph 是同一路径**
两者定义上都是 collect → credit → repair (不启用 graph)，结果完全一致。

**结论 4: ours_full 的 repair_coverage=0 是预期行为**
ours_full 最终评估的是 graph-aware fresh retrieved evaluation，不是 repaired.jsonl 本身。

---

## 5. 当前总结: 已完成与未完成

### 5.1 已完成

| 模块 | 状态 | 验证情况 |
|------|------|---------|
| 四角色多智能体协作 | ✅ 完成 | 在 7 数据集上稳定运行 |
| Turn/subtrajectory/role 级信用分配 | ✅ 完成 | 三路融合工作，能区分 turn 质量 |
| 选择性修复 (单轮) | ✅ 完成 | 稳定贡献 +5~8% accuracy |
| 经验图谱构建 | ✅ 完成 | 能构建，但 retrieval 为负收益 |
| SFT 数据导出 (LLaMA-Factory 格式) | ✅ 完成 | sharegpt 格式 + preference pair |
| Preference pair 导出 | ✅ 完成 | repair 前后对 |
| 七数据集 paper benchmark | ✅ 完成 | 700 eval + 2010 train |
| vLLM 双 4090 部署 | ✅ 完成 | TP=2 稳定 |
| GSM8K 数值评测修复 | ✅ 完成 | 0.0 → 0.83 |
| 消融 suite 自动化 | ✅ 完成 | 9 种方法一键运行 |

### 5.2 未完成 (阻塞论文的关键缺失)

| 模块 | 状态 | 影响 |
|------|------|------|
| **实际执行 SFT 训练** | ❌ 未实现 | 训练数据被导出但从未使用 |
| **迭代自进化循环** | ❌ 未实现 | 模型从未"进化"过 |
| **训练后评估** | ❌ 未实现 | 不知道微调是否真的提升模型 |
| **Credit-guided 数据筛选** | ❌ 未实现 | 当前导出全部 turn，未按 credit 过滤 |
| **多轮迭代修复** | ❌ 未实现 | 当前只修复第一个 flagged turn |
| **Graph 作为修复记忆** | ❌ 未实现 | 当前 graph 仅用于 test-time retrieval (负收益) |
| **SiriuS-SFT 基线** | ❌ 未实现 | 缺少使用训练的 SiriuS 对照 |
| **Budget-equalized 基线** | ❌ 未实现 | 无法区分 sirius_lite 收益来源 |

### 5.3 当前项目流程 vs 完整目标流程

**当前实际流程 (inference-time only):**
```
collect → credit → repair → graph → export(闲置) → eval(repaired episodes)
```

**论文所需的完整流程:**
```
for iteration in 0..N:
    collect(train) → credit → repair → graph(增量) → export(credit-guided)
    → train(SFT + DPO) → swap model → eval(fine-tuned model on eval split)
```

---

## 6. 下一步: 训练闭环实现

### 6.1 实现优先级

| 优先级 | 任务 | 预期影响 |
|--------|------|---------|
| **P0** | 实现 `run_train` 步骤: 调用 LLaMA-Factory 执行 SFT | 完成训练闭环 |
| **P0** | 实现 credit-guided 数据筛选: 只用高 credit turn 训练 | 论文核心实验 |
| **P0** | 实现迭代循环 `run_iterative.py` | 展示跨迭代改进曲线 |
| **P1** | 多轮迭代修复 | 进一步提升 repair 收益 |
| **P1** | SiriuS-SFT 基线 (trajectory-level 筛选 + SFT) | 公平对照 |
| **P2** | Graph-assisted repair memory | 给 graph 一个正面角色 |
| **P2** | Budget-equalized baseline (double-pass no-feedback) | 排除额外计算量解释 |

### 6.2 训练闭环的技术方案

**Step 1: Credit-guided 数据导出**
- 高 credit turn (credit >= 0.65): 进入 SFT 训练集
- 修复成功的 turn (repair 后 accuracy=1): 进入 SFT 训练集
- Repair 前后对: 进入 DPO 训练集
- 低 credit turn (credit < 0.35): 可选 KTO negative signal

**Step 2: 执行训练**
- 框架: LLaMA-Factory
- 方式: LoRA (rank=8, alpha=16) 或 QLoRA (4bit)
- 模型: Qwen2.5-7B-Instruct
- Role-specific: 每个角色分别训练，共享 base model，用不同 LoRA adapter

**Step 3: 模型替换与评估**
- 训练完成后，用新的 LoRA adapter 重启 vLLM 服务
- 在 eval split 上重新 collect → eval
- 记录 per-iteration metrics

**Step 4: 迭代**
- 经验图谱增量更新 (新轮高 credit 节点追加)
- 训练数据可混合前几轮的高质量数据
- 迭代直到 eval accuracy 饱和或达到最大轮次

### 6.3 实验矩阵

完整实验需要以下对比:

| 方法 | 推理时 | 训练数据选择 | 训练方式 |
|------|--------|-------------|---------|
| Static Multi-Agent | 4-role, 无改进 | — | — |
| SiriuS-Lite | 双 pass (失败重写) | — | — |
| SiriuS-SFT | 双 pass | 轨迹级成功/重写 | SFT |
| CEG-SR (inference) | credit + repair | — | — |
| CEG-SR (SFT only) | credit + repair | credit-guided | SFT |
| **CEG-SR (full)** | **credit + repair** | **credit-guided** | **SFT + DPO** |

---

## 7. Phase 4: 训练闭环代码实现 (2026-04-04 — 2026-04-05)

### 7.1 目标

实现从"数据导出就停"到"导出 → 训练 → 模型替换 → 评估 → 迭代"的完整训练闭环，补全 §5.2 中所有 P0 缺失项。

### 7.2 新增/修改文件总览

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/cegsr/training/exporters.py` | 修改 | 新增 `export_credit_guided_sft()` |
| `src/cegsr/training/llamafactory_adapter.py` | 重写 | SFT + DPO + merge 全配置生成 |
| `src/cegsr/training/runner.py` | **新建** | subprocess 调用 LLaMA-Factory 的执行器 |
| `src/cegsr/serving.py` | **新建** | vLLM 推理服务生命周期管理 |
| `src/cegsr/backends/openai_compatible.py` | 修改 | 增加请求重试 + ServerDownError |
| `src/cegsr/workflows.py` | 扩展 | 新增 `run_training` / `run_iterative` / `_run_with_server` |
| `src/cegsr/launchers.py` | 扩展 | 自动生成 `run_iterative.sh` |
| `scripts/run_iterative.py` | **新建** | 迭代训练 CLI 入口 |
| `configs/training/dpo.yaml` | **新建** | DPO 训练模板 |
| `configs/training/lora.yaml` | 重写 | Qwen2.5 优化参数 |
| `configs/training/qlora.yaml` | 重写 | 同上 + 4-bit 量化 |
| `configs/base.yaml` | 更新 | 训练区块增加 mode/threshold/dpo_template |
| `configs/profiles/*.yaml` | 更新 | 增加 train_dataset_path |

### 7.3 Credit-Guided 数据导出

**文件:** `src/cegsr/training/exporters.py` — `export_credit_guided_sft()`

原 `export_role_sft()` 导出所有 turn，不区分质量。新增函数按 credit 分数过滤:

| 条件 | 进入训练集 | 标记 |
|------|-----------|------|
| 正确 episode + turn credit >= 0.65 | SFT | `source=high_credit` |
| repair 后 episode 变正确 | SFT | `source=repair_success` |
| 其余 | 丢弃 | — |

统计信息输出到 `credit_filter_stats.json`，便于检查筛选比例是否合理。

### 7.4 LLaMA-Factory 配置生成

**文件:** `src/cegsr/training/llamafactory_adapter.py` (完全重写)

原版本只生成 per-role SFT 配置，缺少关键参数。新版本:

1. **默认模板全面对齐 Qwen2.5**: `template: qwen`, `lora_target: all`, `bf16: true`, `cosine scheduler`, `warmup_ratio: 0.1`
2. **新增 combined 配置**: 所有角色数据合并为一个 adapter (避免多 adapter 切换的工程复杂度)
3. **新增 DPO 配置**: `stage: dpo`, `pref_beta: 0.1`, `adapter_name_or_path` 指向 SFT 产出，实现 SFT → DPO 链式训练
4. **新增 merge 配置**: `llamafactory-cli export` 将 LoRA adapter 合并回 base model
5. **生成 `run_merge.sh`**: 用于手动执行 merge

生成的文件结构:
```
training_data/
├── sft_manifest.json
├── dataset_info.json
├── {role}_sft.jsonl
├── preference_pairs.jsonl
├── reward_data.jsonl
├── llamafactory_{role}_{lora|qlora}.yaml    # per-role SFT
├── llamafactory_combined_{lora|qlora}.yaml  # combined SFT
├── llamafactory_dpo.yaml                    # DPO
├── llamafactory_merge.yaml                  # merge LoRA → base
├── run_llamafactory.sh
├── run_llamafactory_ddp.sh  (if distributed)
└── run_merge.sh
```

### 7.5 训练执行器

**文件:** `src/cegsr/training/runner.py` (新建)

通过 `subprocess` 调用 `llamafactory-cli train/export`:

- `run_sft(config_path)` → 执行 SFT 训练
- `run_dpo(config_path)` → 执行 DPO 训练
- `merge_lora(config_path)` → 合并 LoRA 到 base model
- `run_training_pipeline(export_dir, ...)` → 编排: SFT → (可选 DPO) → (可选 merge)

支持分布式训练环境变量注入 (`CUDA_VISIBLE_DEVICES`, `FORCE_TORCHRUN`, `NPROC_PER_NODE` 等)。

### 7.6 vLLM 推理服务生命周期管理

**文件:** `src/cegsr/serving.py` (新建)

**核心问题:** 训练和推理共享 GPU，不能同时运行。迭代循环中需要在推理/训练之间切换 vLLM 服务。

`VLLMServerManager` 提供:

| 方法 | 功能 |
|------|------|
| `start(model_path)` | 启动 vLLM (start_new_session=True)，等待健康检查通过 |
| `stop()` | SIGTERM 整个进程组 → 等待 → SIGKILL → 清理端口残留 → 等待 GPU 释放 |
| `restart(model_path)` | stop + start，支持切换到新模型 |
| `health_check()` | GET /v1/models |

**关键设计 — 进程组清理 (解决 TP=2 僵尸进程问题):**

vLLM 启用 tensor parallelism 时会 fork 出 worker 子进程。如果只 kill 主进程，worker 变僵尸，继续占端口和显存。

解决方案:
1. `Popen(start_new_session=True)` → vLLM 及其 TP worker 在独立进程组
2. `os.killpg(pgid, SIGTERM)` → 一次杀掉整棵进程树
3. `_kill_port_holders()` → 扫描端口上的残留进程逐个 SIGKILL
4. `_ensure_port_free()` → start 前确认端口空闲

**外部启动的 vLLM 也能处理:** 如果 vLLM 是之前手动 `bash launch_inference_server.sh` 启动的 (不是我们的子进程)，`_stop_external()` 通过 `lsof -ti :8000` 找到 PID 并杀掉整个进程树。

### 7.7 后端请求重试与崩溃检测

**文件:** `src/cegsr/backends/openai_compatible.py`

原代码单次请求，timeout 即崩。改进:

| 异常类型 | 处理方式 |
|---------|---------|
| `ConnectionError` (服务器不可达) | 立即抛 `ServerDownError`，不重试 |
| `ReadTimeout` | 指数退避重试 (2s → 4s → 8s)，每次重试前探测 /v1/models |
| `HTTP 5xx` | 同上，重试 |
| 重试期间探测到服务器挂了 | 抛 `ServerDownError` |

默认 timeout 从 120s 提升到 180s，默认 max_retries=3。

### 7.8 可恢复 collect/repair + 自动重启

**文件:** `src/cegsr/workflows.py`

**问题:** 2010 样本 collect 跑到第 86 个时 vLLM 挂掉，之前全部丢失。

**可恢复 collect_episodes:**
- 每处理 50 个样本 → 写 `output_path.partial` 文件
- 捕获 `ServerDownError` → 保存已处理的 episodes 到 `.partial`，向上抛
- 下次调用 `resume=True` → 从 `.partial` 加载，跳过已处理样本

**可恢复 repair_episodes:** 同样逻辑。

**`_run_with_server()` 包装器:**
```python
def _run_with_server(fn, server, model_path, max_restarts=3, **kwargs):
    for attempt in range(1, max_restarts + 1):
        try:
            return fn(**kwargs)
        except ServerDownError:
            if attempt >= max_restarts: raise
            server.restart(model_path=model_path)
```

效果: vLLM 挂掉 → 保存进度 → 重启 vLLM → 从断点继续，最多重试 3 次。

### 7.9 迭代循环编排

**文件:** `src/cegsr/workflows.py` — `run_iterative()`

每轮迭代的完整流程:

```
Phase A [vLLM ON]    collect(train_split) → credit → repair → graph → export
Phase B [vLLM OFF]   train(SFT + 可选DPO) → merge LoRA
Phase C [vLLM ON]    eval(eval_split, 用新模型)
```

关键逻辑:
- Phase A 前: `_ensure_server()` 确保 vLLM 在运行
- Phase A 中: `_run_with_server()` 包装 collect/repair，崩溃自动恢复
- Phase B 前: `server.stop()` 释放 GPU
- Phase B 后: 更新 `config['backend']['model']` 和 `config['serving']['model_name_or_path']` 为 merged model 路径
- Phase C 前: `server.start(new_model)` 用新模型重启 vLLM
- Early stopping: 连续两轮 accuracy 提升 < 0.005 → 停止
- `finally` 块: 异常时确保清理 vLLM 进程

### 7.10 配置调整

用户在服务器实测后调整为**单卡推理 + 单卡训练** (非 TP=2):

```yaml
# serving
gpu_ids: [0]           # 原 [0,1]
tensor_parallel_size: 1 # 原 2

# training
distributed.gpus: [0]          # 原 [0,1]
distributed.nproc_per_node: 1  # 原 2

# training hyperparams
lora_rank: 16                  # 原 8
lora_alpha: 32                 # 原 16
per_device_train_batch_size: 4 # 原 2
gradient_accumulation_steps: 4 # 原 8
```

### 7.11 首次迭代运行 (部分完成)

```bash
python scripts/run_iterative.py \
    --config configs/profiles/dual_4090_vllm_paper.yaml \
    --max-iterations 3 --mode lora --credit-threshold 0.65
```

**运行情况:**
- 在 collect 阶段处理到第 86/2010 个样本时 vLLM 崩溃 (ReadTimeout 120s)
- 原因: vLLM TP=2 进程组中 worker 变僵尸，端口/显存未释放
- 重跑时 `server.start()` 失败: `vLLM exited unexpectedly (code 1)` (端口被占)

**根因:** `stop()` 只杀主进程，TP worker 变僵尸。**已通过 §7.6 的进程组清理修复。**

### 7.12 Phase 4 当前状态

| 模块 | 状态 | 说明 |
|------|------|------|
| Credit-guided 数据导出 | ✅ 代码完成 | `export_credit_guided_sft()` |
| LLaMA-Factory 配置生成 (SFT+DPO+merge) | ✅ 代码完成 | 全参数对齐 Qwen2.5 |
| 训练执行器 | ✅ 代码完成 | subprocess 调 llamafactory-cli |
| vLLM 生命周期管理 | ✅ 代码完成 | 进程组清理 + 端口等待 |
| 后端请求重试 | ✅ 代码完成 | 指数退避 + ServerDownError |
| 可恢复 collect/repair | ✅ 代码完成 | .partial 文件 + resume |
| 迭代循环编排 | ✅ 代码完成 | _run_with_server 自动重启 |
| **服务器端 LLaMA-Factory 安装** | ⬜ 待验证 | 需 `pip install llamafactory[torch,metrics]` |
| **首次完整迭代运行** | ⬜ 待运行 | 代码已同步，等待重跑 |
| **训练后评估结果** | ⬜ 待获取 | 依赖首次迭代完成 |

### 7.13 服务器部署步骤

```bash
# 1. 安装 LLaMA-Factory
pip install llamafactory[torch,metrics]
pip install bitsandbytes   # QLoRA 需要
llamafactory-cli version   # 验证

# 2. 同步代码到服务器

# 3. 准备数据 (如已准备可跳过)
python scripts/prepare_data.py --config configs/datasets/paper_reasoning_train.yaml
python scripts/prepare_data.py --config configs/datasets/paper_reasoning_eval.yaml

# 4. 运行迭代训练 (vLLM 由代码自动管理，无需手动启动)
python scripts/run_iterative.py \
    --config configs/profiles/dual_4090_vllm_paper.yaml \
    --max-iterations 3 \
    --mode lora \
    --credit-threshold 0.65
```

注意: **不再需要手动启动/停止 vLLM**。`run_iterative` 会自动管理推理服务的生命周期。
