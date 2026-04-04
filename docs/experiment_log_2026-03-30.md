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
