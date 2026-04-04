# CEG-SR 研究方向文档

> **CEG-SR: Causal Experience Graph with Selective Repair for Self-Evolving Multi-Agent Systems**
>
> 本文档面向已熟悉 multi-agent reasoning、self-improvement、process supervision 方向的研究者，用于在方法设计、代码实现和论文写作之间保持一致。
>
> 最后更新: 2026-04-04

---

## 一、SiriuS 核心方法回顾

SiriuS ([Zhao et al., NeurIPS 2025][1]) 提出了一套 bootstrapped multi-agent self-improvement 循环:

1. **轨迹收集**: 多 agent 协作求解，得到完整 interaction trajectory。
2. **二分过滤**: 成功轨迹 (R > epsilon) 直接保留; 失败轨迹进入增强管线。
3. **失败增强**: 外部模型生成 feedback → 原 agent 重生成 → rephrase 消除 feedback 痕迹 → 级联下游 agent。
4. **Role-specific SFT**: 将收集到的数据按角色分组，通过 OpenAI Fine-tuning API 微调各 agent。
5. **迭代**: 用微调后的模型替代原模型，重复步骤 1-4。

论文在 Problem-Solving (College Physics / Chemistry)、Actor-Critic、Competitive 三类多智能体结构上进行了实验，基线包括 single-agent、STaR、CoMM、TextGrad。

### SiriuS 仓库实现

仓库 ([zou-group/sirius][2]) 的实际代码:
- 三个任务目录 (`Problem_solving/PhyChem/`, `Actor_Critic/`, `Competitive/`)，各自有一套手工串联的脚本: `get_a_sol.py` → `merge.py` → `get_b_feedback.py` → `get_c_regenerate.py` → `get_finetune_data.py` → `fine_tune.py`
- `agent.py` 直接绑定 `openai.OpenAI()`，`fine_tune.py` 直接提交 OpenAI SFT job
- 循环是手工的 (每次迭代重跑全部脚本)，无自动化迭代框架
- 仅支持闭源模型 (gpt-3.5-turbo, gpt-4o-mini)，无开源模型实验

---

## 二、SiriuS 及现有方法的关键不足

### 2.1 信用分配: 整条轨迹粒度，无法定位关键 turn

SiriuS 的核心筛选逻辑是"成功轨迹整体保留，失败轨迹整体增强"。论文自己也承认多智能体优化的难点在于 task-level reward 难以分配到具体 agent 的中间决策。但实际方案只是做了近似: 成功轨迹中每个 agent 的每个 turn 被同等对待为"好数据"。

**这意味着:**
- 成功轨迹中可能存在的冗余 turn、误导性推理都被当作正样本训练
- 失败轨迹中可能正确的前几步推理被完全丢弃
- 训练数据的信噪比不受控

**近期相关进展:**
- **MALT** ([Putta et al., 2024][3], COLM 2025): 构建 multi-agent search tree，用 value iteration 把 reward 回传到各 role 的具体 action，生成 per-agent preference pair。但仅限固定 3-agent (Generator-Verifier-Refiner) pipeline，且依赖 tree search 的计算开销。
- **CCPO** ([2026][4]): 用反事实轨迹 (移除某 agent 后的模拟结果) 估计每个 agent 的边际贡献。但反事实方法随 agent 数量线性增长，且需要 N 次额外 rollout。
- **HCAPO** ([2026][5]): 用 LLM 自身做 post-hoc critic，对 step-level Q-value 做 hindsight 修正。在单 agent 长时序任务上有效，但未扩展到多 agent 协作场景。

### 2.2 失败利用: 整段重写，丢失有价值的局部推理

SiriuS 对失败轨迹的处理是: 外部 feedback → 整段重生成 → rephrase。问题:
- 失败轨迹的前几步可能完全正确，整段重写把它们一起冲掉
- rephrase 容易产生 rationalization (事后编造推理过程)
- 论文在 Actor-Critic 部分也承认: judgment agent 判断不准时，会把正确答案拉去错误修改

**近期相关进展:**
- **Process supervision** ([Lightman et al., 2023][6]): step-level verification 比 outcome supervision 更有效
- **AgentPRM** ([2025][7]): 用 Monte Carlo rollout 为 LLM agent 的每一步生成 process reward target，3B 模型击败 GPT-4o
- 这些工作的共同启示: 失败轨迹不应"整体丢弃或整体重写"，而应被切成可诊断、可修复的局部单元

### 2.3 训练方式: 纯 SFT，无偏好优化

SiriuS 仅用 supervised fine-tuning (在成功/修复后的轨迹上做 next-token prediction)。这有两个根本性限制:
- SFT 无法利用"什么是坏的"这一信号 — 只能模仿好数据，不能学会避免坏数据
- 当好数据量不足或含噪声时，SFT 容易过拟合到表面 pattern

**近期趋势:**
- MALT 已从 search tree 中提取 preference pair，用于 DPO/offline RL
- CollabUIAgents ([2025][8]) 使用 LLM-assigned process reward 合成 preference data，role-free agent 在 7B 规模上达到 GPT-4o 水平
- GRPO、PPO 在 reasoning task 上的成功表明: RL 信号 (尤其是 process-level) 比纯 SFT 更能推动模型推理能力的提升

### 2.4 开源模型: 完全缺失

SiriuS 所有实验依赖 OpenAI API 做 fine-tuning (gpt-3.5-turbo / gpt-4o-mini)。这意味着:
- 无法精细控制训练过程 (学习率、数据配比、LoRA rank 等)
- 无法做 preference optimization (DPO/ORPO) — OpenAI API 只支持 SFT
- 实验不可完全复现
- 无法在学术环境中做消融实验

### 2.5 小结: SiriuS 留下的研究空间

| 维度 | SiriuS | 空缺 |
|------|--------|------|
| 信用分配粒度 | 轨迹级 (成功/失败二分) | turn-level / subtrajectory-level |
| 失败利用方式 | 整段重写 | 选择性局部修复 |
| 训练信号 | 纯 SFT | SFT + Preference (DPO) + Process Reward |
| 经验复用 | 无持久化记忆 | 经验图谱 (可检索、可跨迭代积累) |
| 模型生态 | 闭源 OpenAI API | 开源 Qwen/LLaMA + LoRA |
| 迭代能力 | 手工脚本循环 | 自动化 iterative loop |

---

## 三、CEG-SR 方法设计

### 3.1 核心思想

**将"经验"的基本单位从整条 trajectory 重定义为"带信用分数的子轨迹单元"。** 每条轨迹被分解为 turn-level 和 subtrajectory-level 的单元，各自携带:
- 角色标签、输入状态、输出内容
- 融合后的信用分数 (outcome + verifier + dependency 三路加权)
- 是否被下游引用/修改
- 错误类型标签

然后围绕三个核心机制构建 self-improvement 循环:

```
                    ┌──────────────────────────────┐
                    │     Iterative Loop (T轮)      │
                    └──────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
 ┌─────────────┐         ┌──────────────┐         ┌──────────────┐
 │ Fine-grained │         │  Selective    │         │   Credit-    │
 │    Credit     │  ───►  │   Repair     │  ───►   │   Guided     │
 │  Assignment   │         │ (局部修复)    │         │   Training   │
 └─────────────┘         └──────────────┘         └──────────────┘
        │                         │                         │
        │                 Experience Graph                  │
        │              (修复记忆 + 错误模式库)                │
        └─────────────────────────┼─────────────────────────┘
                                  │
                          下一轮模型替换
```

### 3.2 细粒度信用分配 (Fine-grained Credit Assignment)

对每条轨迹的每个 turn，计算三路信号的加权融合:

**信号 1: Outcome Credit**
- 直接来自最终任务结果 (正确/错误)
- 粒度: episode-level，但作为 baseline 锚定其他信号

**信号 2: Verifier Credit**
- 利用 verifier agent 的输出 (VERDICT: correct/incorrect, Score: 0-1)
- 对非 verifier turn: 基于 verifier 结论做启发式调整
  - 如果 verifier 判 correct: solver +0.1, planner -0.05
  - 如果 verifier 判 incorrect: solver -0.15, summarizer -0.15
  - 如果 turn 中包含犹豫词 ("not sure", "maybe"): 上限 0.45
  - 如果 solver 给出了错误 answer: 上限 0.42
- 对 verifier turn 本身: 从其 Score 文本中提取，但 cap correct 到 0.85、incorrect 到 0.35 (防止 verifier 自我膨胀)

**信号 3: Dependency Credit**
- 基于 turn 之间的依赖图结构
- 被后续 turn 引用且对最终结果有贡献的 turn 获得更高信用

**融合方式:**
```
credit(turn) = w_outcome * outcome + w_verifier * verifier + w_dependency * dependency
```
当前权重: outcome=0.35, verifier=0.4, dependency=0.25

### 3.3 选择性修复 (Selective Repair)

与 SiriuS 的整段重写不同，CEG-SR 只修复被 credit 标记为有问题的局部 span:

**修复触发条件:**
- Turn credit < turn_threshold (0.55)
- 且满足以下之一:
  - Verifier signal < verifier_issue_threshold (0.6)
  - 该 episode 整体失败，且该 turn 属于 solver 或 summarizer (放宽触发)

**修复过程:**
1. 保留高 credit 的 prefix (作为锚定上下文)
2. 从第一个 flagged turn 开始 re-run suffix
3. Re-run 时注入 repair context: repair_reason、preserved_context
4. 重新计算 final_prediction 和 metrics

**当前限制与改进方向:**
- 当前只修复第一个 flagged turn → 需要扩展为多轮迭代修复
- 当前不利用 experience graph → 需要在修复时注入相关成功经验作为 repair memory

### 3.4 经验图谱 (Causal Experience Graph)

**定位调整:** 基于前期实验，experience graph 用于 test-time fresh evaluation 时效果为负。因此将其重新定位为 **repair memory + error pattern library**，而非 test-time oracle。

**图结构:**
- **节点**: 高 credit 的 subtrajectory unit (credit >= min_credit)
  - 属性: role, text, credit score, is_repaired, source_question, dataset_name
  - Embedding: question + role + response 的联合表示
- **边**: temporal dependency, causal support, same-error-type, same-role-transfer

**使用场景:**
1. **修复时的记忆支持**: 当 selective repair 修复某个低 credit turn 时，从图中检索同 role、同 dataset、相似 question 的高 credit 节点，作为 repair prompt 的参考
2. **错误模式分析**: 跨迭代积累的错误节点，帮助识别模型的系统性弱点
3. **跨迭代知识迁移**: 前几轮积累的成功经验在后续迭代的修复阶段可复用

### 3.5 信用引导的训练数据筛选与微调 (Credit-Guided Training)

**这是当前项目最关键的缺失模块，也是 CEG-SR 区别于 SiriuS 的核心实验支柱。**

#### 数据筛选策略

不是所有 episode 的所有 turn 都进入训练。按 credit 分数筛选:

| 数据来源 | 训练方式 | 筛选条件 |
|---------|---------|---------|
| 高 credit turn (成功轨迹) | Role-specific SFT | credit >= high_threshold |
| 修复成功的 turn | Role-specific SFT | repair 前错误 → repair 后正确 |
| Repair 前后对 | DPO/Preference | old_span (低 credit) vs new_span (高 credit) |
| 低 credit turn (失败轨迹) | KTO negative signal | credit < low_threshold |

#### 训练方式: SFT + DPO 两阶段

**阶段 1: Credit-guided SFT**
- 对每个 role 分别训练
- 训练数据 = 高 credit turn + 修复成功的 turn
- 用 LLaMA-Factory LoRA/QLoRA
- 模型: Qwen2.5-7B-Instruct (推理) / 14B-Instruct (训练)

**阶段 2: Repair-derived DPO (可选增强)**
- 利用 selective repair 产生的 (old_span, new_span) 对
- old_span = rejected, new_span = chosen
- 训练 agent 避免之前犯过的错误

#### 与 SiriuS 训练的对比

| 维度 | SiriuS | CEG-SR |
|------|--------|--------|
| 数据筛选 | 轨迹级二分 (成功=全部保留, 失败=全部重写) | Turn-level credit 过滤，只保留高质量 turn |
| 训练信号 | 纯 SFT | SFT + DPO (repair-derived preference) |
| 数据质量 | 成功轨迹中可能含冗余/误导 turn | 只用 credit > threshold 的 turn |
| 失败利用 | 整段 feedback + regeneration + rephrase | 局部修复成功的 turn 直接入训练 |
| 训练框架 | OpenAI API (不可控) | LLaMA-Factory + LoRA (完全可控) |

### 3.6 迭代自进化循环 (Iterative Self-Improvement Loop)

完整的单次迭代:
```
Step 1: Collect    — 用当前模型跑多智能体轨迹 (train split)
Step 2: Credit     — 对每条轨迹做 turn/subtrajectory/role 级信用分配
Step 3: Repair     — 对低 credit turn 做选择性修复 (可迭代多轮)
Step 4: Graph      — 更新经验图谱 (增量式，高 credit 节点入图)
Step 5: Export     — 导出 credit-guided 训练数据 (SFT + DPO)
Step 6: Train      — 调用 LLaMA-Factory 执行 role-specific SFT (+ optional DPO)
Step 7: Swap       — 用微调后的 LoRA adapter 替换当前推理模型
Step 8: Evaluate   — 在 eval split 上评估微调后的模型
```

多轮迭代:
```
for iteration in 0..N:
    if iteration > 0:
        load fine-tuned model from previous iteration
    run Steps 1-8
    if eval_accuracy saturates:
        break
```

**关键设计决策:**
- 推理和训练可以用不同大小的模型 (如 7B 推理、14B 训练)，但在同一迭代内保持一致
- 经验图谱跨迭代增量更新 (不是每轮重建)
- 训练数据可以混合当前轮和前几轮的高质量数据 (经验复用)

---

## 四、与相关工作的对比定位

| 方法 | 信用分配 | 失败利用 | 训练方式 | 经验复用 | 开源模型 |
|------|---------|---------|---------|---------|---------|
| SiriuS | 轨迹级 | 整段重写 | SFT | 无 | 否 |
| MALT | Value iteration (search tree) | Tree search prune | SFT + preference | 无 | 是 |
| CCPO | 反事实 (N次rollout) | 无修复 | RL (PPO) | 无 | 是 |
| HCAPO | Hindsight Q-value | 无修复 | RL (GRPO) | 无 | 是 |
| AgentPRM | MC rollout PRM | 无修复 | RL (actor-critic) | 无 | 是 |
| **CEG-SR** | **Turn-level 三路融合** | **选择性局部修复** | **SFT + DPO** | **经验图谱** | **是** |

**CEG-SR 的差异化定位:**
- 相比 SiriuS: 更细粒度的信用分配 + 选择性修复 + 开源训练闭环
- 相比 MALT: 不依赖 search tree (计算开销更小)，且有经验图谱做跨迭代记忆
- 相比 CCPO: 不需要 N 次反事实 rollout，信用分配融合多种信号
- 相比所有已有工作: **唯一将 fine-grained credit assignment 与 selective repair 和 experience graph 三者结合的方法**

---

## 五、实验设计

### 5.1 硬件与模型

| 资源 | 配置 |
|------|------|
| GPU | RTX 4090 x2 |
| 推理后端 | vLLM (tensor_parallel_size=2) |
| 推理模型 | Qwen2.5-7B-Instruct |
| 训练模型 | Qwen2.5-7B-Instruct (LoRA/QLoRA) |
| 训练框架 | LLaMA-Factory |
| 迭代轮次 | 3-5 轮 |

### 5.2 数据集 (七数据集 Paper Benchmark)

| 数据集 | 类型 | Eval 数量 | Train 数量 |
|--------|------|----------|-----------|
| college_physics | MCQ (MMLU) | 100 | 5 (dev only) |
| college_chemistry | MCQ (MMLU) | 100 | 5 (dev only) |
| pubmed_qa | Yes/No/Maybe | 100 | 400 |
| gsm8k | 数值计算 | 100 | 400 |
| commonsense_qa | MCQ | 100 | 400 |
| ai2_arc | MCQ | 100 | 400 |
| boolq | Yes/No | 100 | 400 |
| **Total** | | **700** | **2010** |

注意: college_physics / college_chemistry 的训练集目前仅有 dev split (各 5 条)，后续可考虑扩充。

### 5.3 主实验表 (论文 Table 1)

**目标:** 展示 CEG-SR 迭代训练后的模型性能，横向对比基线方法。

| 方法 | 描述 | 是否涉及训练 |
|------|------|-------------|
| Single Agent | 单 agent 直接回答 | 否 |
| Static Multi-Agent | 4-role 协作，无自进化 | 否 |
| SiriuS-Lite | 整段失败重写 (无训练) | 否 |
| SiriuS-SFT | 整段失败重写 + 轨迹级 SFT | 是 |
| CEG-SR (inference) | Credit + Repair (无训练) | 否 |
| **CEG-SR (full)** | **Credit + Repair + Training loop** | **是** |

**关键对比:**
- SiriuS-Lite vs CEG-SR (inference): 消除训练变量，纯比推理时修复策略
- SiriuS-SFT vs CEG-SR (full): 完整 self-improvement 循环对比

### 5.4 消融实验 (论文 Table 2)

| 消融变体 | Credit | Repair | Training | Graph |
|---------|--------|--------|----------|-------|
| Trajectory-level credit + SFT | 轨迹级 | 无 | 是 (全数据) | 无 |
| Fine-grained credit + SFT (no repair) | Turn-level | 无 | 是 (高credit) | 无 |
| Fine-grained credit + Repair (no training) | Turn-level | 是 | 否 | 无 |
| Fine-grained credit + Repair + SFT | Turn-level | 是 | 是 | 无 |
| **Full CEG-SR** | **Turn-level** | **是** | **是** | **修复记忆** |

**消融要回答的问题:**
1. Turn-level credit vs Trajectory-level credit → 数据筛选质量对训练效果的影响
2. Credit-guided SFT vs 全数据 SFT → 信用引导训练是否优于盲目训练
3. Repair + Training vs Repair only → 训练是否放大了修复效果
4. Graph-assisted repair vs 无 graph repair → 经验图谱对修复质量的贡献

### 5.5 分析实验

1. **迭代曲线**: eval accuracy / repair_success_rate / training_data_quality 随迭代轮次的变化
2. **Per-dataset breakdown**: 不同数据集上各方法的表现差异
3. **Case study**: 3-5 个典型样本的修复前后对比 (含 credit 分布可视化)
4. **Training data quality analysis**: credit-guided 筛选后的数据 vs 全量数据的质量对比

---

## 六、论文叙事框架

**一句话卖点:**
> SiriuS 解决了"多智能体系统能自进化"，CEG-SR 解决"经验库里什么真正有用、怎么局部修复、怎么选择性训练"。

**Introduction 叙事线:**
1. Multi-agent LLM 系统在复杂推理任务上优于单 agent (已有共识)
2. SiriuS 等方法证明了 bootstrapped self-improvement 的可行性
3. 但信用分配太粗 (轨迹级) → 训练数据含噪、失败轨迹被浪费
4. 我们提出 CEG-SR: 细粒度信用分配 + 选择性修复 + 信用引导训练 + 经验图谱
5. 在七数据集 benchmark 上，CEG-SR 的迭代训练显著优于 SiriuS-style SFT

**Method 结构:**
- 3.1 Problem Formulation (multi-agent trajectory, credit, repair)
- 3.2 Fine-grained Credit Assignment
- 3.3 Selective Repair
- 3.4 Credit-guided Training (SFT + DPO)
- 3.5 Causal Experience Graph
- 3.6 Iterative Self-Improvement Loop

**Expected contribution:**
1. 首个将 turn-level credit assignment 与 selective repair 结合的多智能体自进化方法
2. 提出 credit-guided training data selection，证明数据质量比数据数量更重要
3. 将 experience graph 定位为 repair memory (而非 test-time oracle)，给出正面结果
4. 完整的开源 iterative self-improvement 框架，可复现

---

## 参考文献

[1]: https://arxiv.org/abs/2502.04780 "SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning (NeurIPS 2025)"
[2]: https://github.com/zou-group/sirius "zou-group/sirius GitHub"
[3]: https://arxiv.org/abs/2412.01928 "MALT: Improving Reasoning with Multi-Agent LLM Training (COLM 2025)"
[4]: https://arxiv.org/abs/2603.21563 "CCPO: Counterfactual Credit Policy Optimization"
[5]: https://arxiv.org/abs/2603.08754 "HCAPO: Hindsight Credit Assignment for Long-Horizon LLM Agents"
[6]: https://arxiv.org/abs/2305.20050 "Let's Verify Step by Step (ICLR 2024)"
[7]: https://arxiv.org/abs/2502.10325 "AgentPRM: Process Reward Models for LLM Agents"
[8]: https://arxiv.org/abs/2502.14496 "CollabUIAgents: Credit Re-Assignment for Interactive Environment Generalization"
