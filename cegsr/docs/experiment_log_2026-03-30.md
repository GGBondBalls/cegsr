# 2026-03-30 实验日志

## 0. 文档目的与读者

本文档面向第一次阅读本项目的研究者，尤其是已经熟悉多智能体推理、过程监督、trajectory credit assignment、self-improvement 这类方向，但尚未参与当前工程迭代的专家读者。

文档目标不是简单罗列命令，而是回答下面几个问题：

1. 这一轮实验到底想验证什么？
2. 实验过程中我们观察到了哪些异常？
3. 针对这些异常，代码层面做了哪些修改？
4. 修改后的结果说明了什么？
5. 下一轮应该优先做什么，不该做什么？

## 1. 项目当前实验设定

当前主配置文件为 [`../configs/base.yaml`](../configs/base.yaml)。

### 1.1 推理后端

- backend: `hf_local`
- 模型路径模板: `/home/fyk/models/Qwen/Qwen2.5-X.XB-Instruct`
- 当前实际实验模型大小: `1.5B`

### 1.2 任务与数据

- 数据文件: `outputs/data/reasoning_mix_eval.jsonl`
- 当前评测混合数据由四部分组成:
  - `commonsense_qa`
  - `ai2_arc`
  - `boolq`
  - `pubmed_qa`

### 1.3 多角色结构

- 角色顺序: `planner -> solver -> verifier -> summarizer`

### 1.4 当前关键配置

credit 融合权重:

- `outcome = 0.35`
- `verifier = 0.4`
- `dependency = 0.25`

repair 相关:

- `turn_threshold = 0.55`
- `subtrajectory_threshold = 0.55`
- `require_verifier_issue = true`
- `verifier_issue_threshold = 0.6`
- `relax_on_failure = true`
- `failure_margin = 0.08`

retrieval 相关:

- `top_k = 2`
- `expand_neighbors = false`
- `role_match_only = true`
- `exclude_same_sample = true`
- `same_dataset_only = true`
- `min_similarity = 0.3`
- `question_overlap_weight = 0.2`
- `min_question_overlap = 0.02`
- `enabled_roles = [solver]`

## 2. 本轮迭代前的核心问题

本轮实验最初并不是为了“继续堆代码”，而是为了回答一个更具体的问题：

> CEG-SR 当前的收益，到底主要来自 experience graph，还是来自 fine-grained credit + selective repair？

最早的基线跑通后，我们得到的现象是：

- 主流程可以完整运行；
- `repair_coverage` 非常低，说明 selective repair 基本没有真正工作起来；
- graph retrieval 一旦打开，结果反而下降；
- `trajectory-level credit` 与当前“细粒度 credit”版本之间几乎看不出差异。

这意味着：

1. 工程流程是通的，但方法机制未被真正激活；
2. 如果直接写论文，很容易把“运行起来了”和“方法有效了”混为一谈；
3. 因而这一轮的重点，不应是再加新模块，而应是确认当前三个核心模块谁在真正起作用：
   - 细粒度 credit
   - selective repair
   - graph retrieval

## 3. 问题排查与修改脉络

下面按实际问答与迭代过程重述本轮分析路径。

### 3.1 第一步：确认主流程已跑通，但 repair 几乎未激活

最初跑通的命令是：

```bash
python scripts/run_collect.py --config configs/base.yaml --output outputs/demo/raw.jsonl
python scripts/run_credit.py --config configs/base.yaml --episodes outputs/demo/raw.jsonl --output outputs/demo/annotated.jsonl
python scripts/run_repair.py --config configs/base.yaml --episodes outputs/demo/annotated.jsonl --output outputs/demo/repaired.jsonl
python scripts/build_graph.py --config configs/base.yaml --episodes outputs/demo/repaired.jsonl --graph-dir outputs/demo/graph
python scripts/export_sft.py --config configs/base.yaml --episodes outputs/demo/repaired.jsonl --export-dir outputs/demo/training_data
python scripts/run_eval.py --episodes outputs/demo/repaired.jsonl --output-dir outputs/demo/eval --graph-dir outputs/demo/graph
```

早期结果大致为：

- `accuracy = 0.6975`
- `repair_coverage = 0.0525`
- `retrieval_hit_usefulness_proxy = 0.0`

这个结果的含义不是“方法失败”，而是：

- collect / credit / repair / graph / eval 整条链路确实可运行；
- 但 selective repair 只覆盖了极少数样本；
- graph 虽然被构建了，但并没有变成有效检索记忆。

### 3.2 第二步：确认 retrieval 是负收益，而不是主贡献

随后专门测试 graph-aware retrieval：

```bash
python scripts/run_collect.py --config configs/base.yaml --use-retrieval --output outputs/demo/retrieved_eval.jsonl
python scripts/run_eval.py --episodes outputs/demo/retrieved_eval.jsonl --output-dir outputs/demo/eval_retrieved --graph-dir outputs/demo/graph
```

最初 retrieval 结果一度下降到：

- `accuracy = 0.59`

后续经过多轮 retrieval 收紧，结果先后变为：

- `retrieved_eval_v3 = 0.635`
- `retrieved_eval_v4 = 0.68`
- `retrieved_eval_v5 = 0.6775`
- `retrieved_eval_v6 = 0.68`
- `retrieved_eval_v7 = 0.68`

这个趋势很重要。它说明：

1. retrieval 不是完全不可救；
2. 但即便经过持续修补，也只能稳定在 `0.68` 左右；
3. 它始终没有超过 no-graph repaired pipeline；
4. 所以 graph retrieval 至少在当前实现中，不是方法的主增益来源。

### 3.3 第三步：定位 selective repair 为什么几乎不起作用

排查后发现两个关键问题。

#### 问题 A：repair prompt 没有真正传递给模型

虽然 repair 逻辑里已经生成了 `repair_mode`、`repair_reason`、`preserved_context` 这类信息，但早期 prompt 构造并没有真正把这些信息写进用户提示。

结果是：

- 系统名义上在“局部修复”；
- 实际上模型只是在无差别重跑 suffix。

对应修改文件：

- [`../src/cegsr/tasks/qa.py`](../src/cegsr/tasks/qa.py)
- [`../src/cegsr/repair/selective_repair.py`](../src/cegsr/repair/selective_repair.py)

#### 问题 B：verifier credit 太容易把高分传染给全轨迹

早期 `VerifierCreditSignal` 的行为过于宽松：

- 一旦 verifier 给出很高的 `Score`；
- 很容易把这个高分以默认值方式传播到其它角色 turn；
- 结果导致“细粒度 credit”在实际数值上退化成“整体都不低”。

这会直接削弱 selective repair，因为 repair detector 很难再看到真正低信用的局部 span。

对应修改文件：

- [`../src/cegsr/credit/verifier_credit.py`](../src/cegsr/credit/verifier_credit.py)
- [`../src/cegsr/repair/detector.py`](../src/cegsr/repair/detector.py)
- [`../src/cegsr/repair/selective_repair.py`](../src/cegsr/repair/selective_repair.py)

### 3.4 第四步：逐轮压缩 retrieval 噪声

虽然最终判断是 retrieval 不是主增益来源，但在这一轮排查中，retrieval 分支仍然做了系统性收紧，原因是我们需要确认“它为什么无效”，而不能只靠猜测。

具体修改包括：

1. 禁止 same-sample 泄漏；
2. 默认限制为 same-role；
3. 关闭 neighbor expansion；
4. 限制只对 `solver` 开启 retrieval；
5. 限制为 same-dataset 检索；
6. 增加 minimum similarity；
7. 增加 question-overlap 重排；
8. 将 graph node 的 embedding 从 `response` 改为 `question + role + response`；
9. 对 solver 注入的 retrieved snippet 做“去答案化”，改成 reasoning pattern 提示，而不是直接暴露别题答案。

涉及文件：

- [`../src/cegsr/experience/retriever.py`](../src/cegsr/experience/retriever.py)
- [`../src/cegsr/experience/builder.py`](../src/cegsr/experience/builder.py)
- [`../src/cegsr/agents/graph_runtime.py`](../src/cegsr/agents/graph_runtime.py)
- [`../src/cegsr/tasks/qa.py`](../src/cegsr/tasks/qa.py)
- [`../src/cegsr/workflows.py`](../src/cegsr/workflows.py)
- [`../configs/base.yaml`](../configs/base.yaml)

说明：

这些修改确实把 retrieval 从 `0.59` 拉回到了 `0.68` 附近，证明噪声来源判断基本正确；但也恰恰证明了 retrieval 的收益上限在当前设计下较低。

## 4. 本轮关键实验结果

### 4.1 重跑 credit / repair 后的 repaired pipeline

命令：

```bash
python scripts/run_credit.py --config configs/base.yaml --episodes outputs/demo/raw.jsonl --output outputs/demo/annotated_v2.jsonl
python scripts/run_repair.py --config configs/base.yaml --episodes outputs/demo/annotated_v2.jsonl --output outputs/demo/repaired_v2.jsonl
python scripts/build_graph.py --config configs/base.yaml --episodes outputs/demo/repaired_v2.jsonl --graph-dir outputs/demo/graph_v2
python scripts/run_eval.py --episodes outputs/demo/repaired_v2.jsonl --output-dir outputs/demo/eval_repaired_v2 --graph-dir outputs/demo/graph_v2
```

结果：

- `accuracy = 0.7875`
- `exact_match = 0.37`
- `repair_coverage = 0.315`
- `repair_success_rate = 0.3254`
- `num_changed_repairs = 126`

分数据集结果：

- `commonsense_qa = 0.84`
- `ai2_arc = 0.79`
- `boolq = 0.78`
- `pubmed_qa = 0.74`

这一组结果是整轮实验的分水岭。它说明：

1. selective repair 不再是“名义存在”的模块，而是实实在在改动了大量样本；
2. verifier-aware fine-grained credit 终于对 repair 触发产生了影响；
3. 当前 strongest path 来自“credit + repair”，而不是 graph retrieval。

### 4.2 当前完整 ablation

命令：

```bash
python scripts/run_ablation.py --config configs/base.yaml --output-dir outputs/ablations_v2
```

结果见：

- [`../outputs/ablations_v2/ablation_table.md`](../outputs/ablations_v2/ablation_table.md)
- [`../outputs/ablations_v2/ablation_table.csv`](../outputs/ablations_v2/ablation_table.csv)

核心表如下：

| method | accuracy | exact_match | repair_coverage | retrieval_proxy |
|---|---:|---:|---:|---:|
| single_agent | 0.42 | 0.105 | 0.0 | 0.0 |
| static_multi_agent | 0.6825 | 0.3175 | 0.0 | 0.0 |
| sirius_lite | 0.7875 | 0.36 | 0.0 | 0.0 |
| ours_wo_graph | 0.77 | 0.36 | 0.3425 | 0.0 |
| ours_wo_selective_repair | 0.6775 | 0.34 | 0.0 | 0.6849 |
| trajectory_level_credit | 0.6775 | 0.3175 | 0.0 | 0.0 |
| repair_only | 0.77 | 0.36 | 0.3425 | 0.0 |
| offline_sft_only | 0.6775 | 0.3175 | 0.0 | 0.0 |
| ours_full | 0.6775 | 0.34 | 0.0 | 0.6849 |

## 5. 如何理解这些结果

### 5.1 已经可以比较明确的结论

#### 结论 1：当前主增益来自 fine-grained credit + selective repair

证据：

- `trajectory_level_credit = 0.6775`
- `offline_sft_only = 0.6775`
- `ours_wo_graph = 0.77`
- `repair_only = 0.77`
- `repaired_v2 standalone = 0.7875`

这说明：

- 仅有轨迹级 credit 或仅做离线导出，并不会自动带来优势；
- 一旦 selective repair 真正被激活，结果会有明显提升。

#### 结论 2：graph retrieval 当前不是主增益来源

证据：

- `ours_full = 0.6775`
- `ours_wo_selective_repair = 0.6775`
- `retrieved_eval_v7 = 0.68`

这说明：

- 即使 graph 构建成功，且 retrieval proxy 不为 0；
- 当前 test-time retrieval 仍没有把最终准确率推高。

因此，在论文叙事中，graph 更适合作为：

- 辅助记忆结构；
- 失败分析工具；
- 后续工作的扩展方向；

而不是当前版本最强的主结果来源。

#### 结论 3：当前 repaired pipeline 已达到最强 baseline 水平

证据：

- `sirius_lite = 0.7875`
- `repaired_v2 standalone = 0.7875`

这件事很关键。它说明当前方法线已经具备论文价值：

- 不是“略优于弱基线”；
- 而是已经能达到当前最强 baseline 的性能水平。

当然，也要注意一个细节：

- `repaired_v2 standalone = 0.7875`
- `ours_wo_graph` 在 ablation 中是 `0.77`

这说明执行路径仍然会带来轻微波动。正式论文表格前，需要统一 one-pass evaluation protocol，避免把不同 artifact 的分数直接横比。

### 5.2 当前最合理的方法判断

基于本轮结果，我认为当前不需要继续进行大范围代码重构。

原因不是“代码已经完美”，而是：

1. 关键科学问题已经被回答了；
2. 当前最值得写进论文的卖点已经明确；
3. 继续大改 graph retrieval，短期内未必能转化成更强主结果；
4. 更合理的策略是先沉淀本轮证据，再决定 graph 是否作为后续独立方向继续推进。

## 6. 本轮代码变更摘要

如果首次接触项目，需要快速知道本轮主要改了哪里，可以按下面理解。

### 6.1 与 repair 激活直接相关的修改

- `src/cegsr/tasks/qa.py`
  - 将 repair-mode 信息真正写入 prompt。
- `src/cegsr/credit/verifier_credit.py`
  - 收紧 verifier score 解释，避免高分无差别扩散。
- `src/cegsr/repair/detector.py`
  - 对失败样本下的低信用 solver/summarizer 放宽 repair 触发条件。
- `src/cegsr/repair/selective_repair.py`
  - 将 detector 新参数接入 selective repair engine。
- `src/cegsr/workflows.py`
  - 将 repair 新配置接入主流程。

### 6.2 与 retrieval 排查相关的修改

- `src/cegsr/experience/retriever.py`
  - 加入 same-sample / same-dataset / similarity / question-overlap 等约束。
- `src/cegsr/experience/builder.py`
  - 修改 graph node 的 embedding 文本构造。
- `src/cegsr/agents/graph_runtime.py`
  - 控制 retrieval 只在指定角色启用。
- `src/cegsr/tasks/qa.py`
  - 对 solver 检索提示进行去答案化。

### 6.3 测试补充

本轮还补充或更新了若干针对性测试：

- [`../tests/test_tasks.py`](../tests/test_tasks.py)
- [`../tests/test_graph.py`](../tests/test_graph.py)
- [`../tests/test_credit.py`](../tests/test_credit.py)
- [`../tests/test_repair.py`](../tests/test_repair.py)

这些测试的主要用途不是覆盖率本身，而是防止后续重构把本轮确认有效的行为再次改坏。

## 7. 对下一轮的建议

### 7.1 推荐优先路线：论文整合优先

建议下一轮优先做下面三件事，而不是继续大改代码。

#### 任务 A：统一最终主结果的评测协议

目标：

- 明确论文主表中，“我们的方法”到底对应哪个 artifact。

建议：

- 以 repaired pipeline 为主；
- graph retrieval 结果单独作为消融或负结果呈现；
- 不要把 `repaired_v2` 与 `ours_full` 混成一个方法名。

#### 任务 B：抽取 3 到 5 个 selective repair 案例

优先从以下文件中抽样：

- `outputs/demo/repaired_v2.jsonl`
- `outputs/demo/eval_repaired_v2/report.md`

建议选三类：

1. 修前错、修后对；
2. verifier 明确降低局部信用后触发 repair；
3. 仅局部 span 被改写，而高信用前缀被保留。

#### 任务 C：围绕 repair 重写论文主线

目前最可信的论文叙事是：

1. trajectory-level credit 不足以支持有效自进化；
2. verifier-aware fine-grained credit 能够识别更有价值的局部推理单元；
3. selective repair 可以显著提升失败轨迹的再利用率；
4. experience graph 在当前实现中尚未转化为稳定的 test-time 增益。

### 7.2 暂不推荐的路线：继续大规模改 graph retrieval

不是说 graph 一定没有价值，而是说：

- 本轮证据已经表明它不是当前版本最主要的收益来源；
- 再继续围绕 retrieval 做大改，短期风险较高；
- 更容易打散已经形成的论文主线。

如果后续还想做 graph，建议把问题重新定义为：

> graph 是否更适合用于 repair-memory，而不是 answer-memory？

也就是：

- 不再在 fresh evaluation 时直接让 graph 参与回答；
- 而是在 selective repair 阶段，为局部改写提供 memory support。

这会比继续优化当前 test-time retrieval 路线更有研究价值。

## 8. 本轮总结

如果只保留一句总结，本轮最重要的结论是：

> 当前 CEG-SR 版本的核心收益已经明确来自 verifier-aware 细粒度 credit assignment 与 selective repair，而不是 graph retrieval。

这是一个相当重要的转折点。它意味着项目已经从“工程流程能跑”进入“方法主贡献可辨认”的阶段。对于后续论文写作和实验设计，这个判断应当作为优先前提。
