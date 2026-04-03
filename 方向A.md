## 一、我对 SiriuS 的核心理解

SiriuS 的基本思想很清楚：先让多智能体系统跑出完整协作轨迹，把高奖励/成功轨迹加入经验库；对失败轨迹，则额外用一个外部 agent 基于题目、原回答和正确答案生成反馈，再让原 agent 重写并“改写成像是直接推理出来的答案”，最后把这些数据拿去做 role-specific 的 SFT。论文里问题求解、actor-critic、competitive 三类设置都用了这个套路，结果在 College Physics、College Chemistry、PubMedQA 等任务上优于 single-agent、STaR、CoMM、TextGrad 等基线。([arXiv][1])

从仓库实现看，它本质上也是这个思路的一个研究原型：README 给出了“生成轨迹→merge→对错误轨迹生成 feedback→regenerate→导出 finetune 数据→调用 OpenAI SFT”的脚本链路；而代码里 `agent.py` 直接绑定 `OpenAI()`，`fine_tune.py` 直接提交每个角色的 OpenAI fine-tuning job，`merge.py` 和 `get_finetune_data.py` 也有明显的任务路径硬编码。这说明仓库更像论文复现实验代码，而不是一个真正可扩展的“自进化多智能体研究框架”。([GitHub][2])

## 二、SiriuS 的关键不足

### 1）信用分配仍然太粗，只到“整条轨迹是否成功”

SiriuS 的核心筛选逻辑仍是：**成功轨迹整体保留，失败轨迹整体进入修复**。论文自己也承认，多智能体优化的难点就在于任务级 reward 很难分配到具体 agent 的中间决策；但它最后采取的是“整条成功轨迹大概率有用”的近似。这样做有效，但也留下了最大的研究空位：**到底是哪一个 turn、哪一段子轨迹、哪个 agent 的消息导致成功或失败，SiriuS 没有真正建模。** ([arXiv][1])

而 2025–2026 的一些工作已经明确朝更细粒度方向走了。MALT 用 multi-agent search tree 加 value iteration，把回报往不同角色回传；CollabUIAgents 用 LLM 给 agent 和 conversation-round 两级打过程奖励；HCAPO 则把 hindsight credit assignment 引入长时程 LLM agent，把轨迹级奖励细化为 step-level credit。SiriuS 相比这些方法，最明显的落点就是“粒度不够细”。([OpenReview][3])

### 2）失败利用方式太“整段重写”，没有 selective repair

SiriuS 对失败轨迹的增强，本质上是：外部 critic 给反馈，原 agent 重生成，然后再 rephrase 成“看起来像自然推理”。这会有两个问题。第一，失败轨迹里常常并不是全错，可能前两步都对，只是第三步出错；整段重写会把原本有价值的局部 reasoning 一起冲掉。第二，这种“整段修复”很容易产生风格漂移和伪造式 rationalization。论文在 actor-critic 部分也提到，judgment agent 判断不准时，会把本来正确的答案错误地拉去修改。([arXiv][1])

这正是 process supervision / verifier 方向已经在解决的问题。比如 Let’s Verify Step by Step 强调 step-level verification 的价值，ThinkPRM 用带推理链的 PRM 做逐步验证，FoVer 则尝试用形式化验证自动合成 step-level error label。它们给你的启发是：**失败轨迹不是“整体丢弃或整体重写”，而是应该被切成可诊断、可修复的局部单元。** ([arXiv][4])

## 三、我建议的创新方向

## 方向 A：把 SiriuS 升级成“因果经验图谱 + 细粒度信用分配 + 选择性修复”

这是我最推荐、也最像一篇完整 CCF-A 主线论文的方向。

核心想法是：**不再把“经验”的基本单位定义为整条 trajectory，而定义为“带信用分数的子轨迹单元”。** 每个 turn 或一段 subtrajectory 都要有：
角色、输入状态、输出内容、被谁引用、是否支持/反驳了后续结论、最终 outcome 贡献、错误类型、可迁移范围。

然后做三件事。

第一，做 **turn-level / subtrajectory-level credit assignment**。
你可以把 credit 设计成一个融合分数：

* outcome 反事实增益：把某一 turn mask 掉或替换成库里的相似候选，看最终答案变化多少；
* verifier/process reward：让 verifier 对每一步给局部正确性和帮助度分；
* hindsight score：借鉴 HCAPO 的 post-hoc hindsight 视角，给成功结局下“这一步是否是更可能导致成功的动作”打分。
  这样可以得到 agent-level、turn-level、subtrajectory-level 三层 credit。([arXiv][7])

第二，做 **selective repair**。
不是整段重写，而是只修 credit 很低、且被 verifier 明确标成错误或误导的局部 span；高 credit 的 span 保留并锁定。失败轨迹于是不会被粗暴“洗掉”，而会变成“局部补丁后的高质量混合轨迹”。这比 SiriuS 的整段 regeneration 更细，也更容易做出可信 ablation。([arXiv][1])

第三，做 **experience graph**。
把高 credit 子轨迹、典型错误模式、修复模板都存成图节点，边包含 temporal dependency、causal support、contradiction、same-error-type、same-role-transfer。推理时不是检索整条成功轨迹，而是检索“当前角色在当前局部状态最相关的几个经验子图”，这会把 SiriuS 从 offline SFT 提升到 **test-time self-evolution**。这条线和 AWM、EvoSC、MUSE、trajectory-informed memory generation 的趋势是同方向的，但你做的是更适合多角色协作 reasoning 的版本。([arXiv][5])

我会把这个方向命名成类似：**CEG-SR：Causal Experience Graph with Selective Repair for Self-Evolving Multi-Agent Systems**。

它的论文卖点会非常集中：
**SiriuS 解决了“有经验库”，我们解决“经验库里什么真正有用、怎么局部复用、怎么局部修”。**


[1]: https://arxiv.org/html/2502.04780 "SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning"
[2]: https://github.com/zou-group/sirius/blob/main/README.md?utm_source=chatgpt.com "README.md - zou-group/sirius"
[3]: https://openreview.net/forum?id=jXP9bgFack "MALT: Improving Reasoning with Multi-Agent LLM Training | OpenReview"
[4]: https://arxiv.org/abs/2305.20050?utm_source=chatgpt.com "[2305.20050] Let's Verify Step by Step"
[5]: https://arxiv.org/html/2409.07429v1 "Agent Workflow Memory"
[6]: https://arxiv.org/abs/2506.01716?utm_source=chatgpt.com "Self-Challenging Language Model Agents"
[7]: https://arxiv.org/pdf/2603.08754 "Hindsight Credit Assignment for Long-Horizon LLM Agents"
[8]: https://arxiv.org/html/2502.14496v2 "Advancing Language Multi-Agent Learning with Credit Re-Assignment for Interactive Environment Generalization"
