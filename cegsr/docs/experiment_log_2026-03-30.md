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

## 9. 2026-03-31：双 RTX4090 + vLLM 运行兼容与服务器基线实验

这一轮工作的目标不是继续细化方法，而是先把后续多轮实验真正放到服务器上稳定跑起来，并把“单卡本地可跑”升级为“双卡服务器可重复执行”的实验基线。

核心判断先写在最前面：

> 目前暂时不需要大范围修改方法代码。当前最值得保留和沉淀的是：在不改变 CEG-SR 核心方法结构的前提下，完成双卡 vLLM 推理兼容、单命令 pipeline 运行、以及同一数据子集上的 raw / annotated / repaired 对照验证。

### 9.1 本轮硬件与模型设置

服务器硬件：

- GPU: `RTX 4090 x2`
- 运行环境：学校服务器个人账号

本轮实际使用的模型设置：

- 推理后端：`vLLM`
- agent 推理模型：`Qwen2.5-7B-Instruct`
- vLLM 并行方式：`tensor_parallel_size = 2`
- 最大上下文：`max_model_len = 4096`
- 并发设置：`max_num_seqs = 16`
- 训练导出配置：仍保留 `Qwen2.5-14B-Instruct` 的双卡 DDP 导出脚本，但这一轮尚未正式启动训练

这与前一轮日志中的 `1.5B + hf_local` 不同。本轮的重点是验证：

1. 双卡服务器是否能稳定承接 collect / repair 主流程；
2. 在不改核心方法逻辑的情况下，运行层切换到 `vLLM + TP=2` 后是否仍然保留 selective repair 收益；
3. 后续是否可以把这套流程作为多轮实验的默认执行入口。

### 9.2 本轮新增的工程优化，不属于方法大改

这一轮代码修改主要集中在“运行兼容层”，而不是方法主干。

#### 改动 A：配置继承与 profile 化

为配置系统增加了 `_base_` 继承能力，使服务器实验不需要复制整份 `base.yaml`，只需在 profile 中覆盖必要字段。

涉及文件：

- `src/cegsr/config/loader.py`
- `configs/profiles/dual_4090_vllm.yaml`

这使得后续可以把：

- 本地 smoke test
- 单卡 HF local
- 双卡 vLLM 推理
- 双卡训练

分别整理为独立 profile，而不是不断手工改同一份配置。

#### 改动 B：模型路径模板统一

将 `X.XB` / `{model_size}` 形式的路径模板统一抽象出来，使 `hf_local`、`vllm`、`sglang` 都能通过同样的 `model_size` 逻辑切换 `7B / 14B`。

涉及文件：

- `src/cegsr/utils/modeling.py`
- `src/cegsr/workflows.py`

这一步很关键，因为后续实验会频繁切换：

- `7B` 作为 agent 推理模型
- `14B` 作为训练目标模型

如果路径切换仍靠手工改字符串，后续实验成本会很高，也容易出错。

#### 改动 C：一键 pipeline 入口

增加了：

- `scripts/run_pipeline.py`

把原先分散的：

- `collect`
- `credit`
- `repair`
- `build_graph`
- `export`
- `eval`

串为一个单命令入口，便于服务器上批量跑实验。

对应主流程函数也增加了：

- `run_pipeline(...)`

涉及文件：

- `scripts/run_pipeline.py`
- `src/cegsr/workflows.py`

#### 改动 D：服务器实验脚本自动生成

增加了：

- `scripts/setup_experiment.py`
- `src/cegsr/launchers.py`

会根据 profile 自动生成：

- `prepare_data.sh`
- `launch_inference_server.sh`
- `run_pipeline.sh`
- `run_ablation.sh`

这一步的意义不是“多写几个脚本”，而是把原本靠记忆和手工拼接的服务器命令固化下来，降低后续多次实验时的认知负担和操作风险。

#### 改动 E：双卡 DDP 训练脚本导出

在训练导出阶段增加了 `distributed` 配置读取，自动生成：

- `run_llamafactory.sh`
- `run_llamafactory_ddp.sh`

后者会写入：

- `CUDA_VISIBLE_DEVICES`
- `FORCE_TORCHRUN`
- `NPROC_PER_NODE`
- `MASTER_ADDR`
- `MASTER_PORT`

涉及文件：

- `src/cegsr/training/llamafactory_adapter.py`

这一步是为后续 `14B` 训练铺路，但本轮还没有真正启动大规模训练。

#### 改动 F：vLLM 兼容性修复

在服务器上第一次启动 vLLM 时，遇到了两个真实问题：

1. 早期启动脚本使用了 `vllm` CLI 参数 `--swap-space 8`，而当前服务器上的 `vllm==0.18.1` 不接受该参数；
2. pipeline 启动前的健康检查访问 `/v1/models` 时未携带鉴权头，导致虽然服务已启动，但探活返回 `401 Unauthorized`。

因此做了两项兼容修复：

- 去掉不兼容的 `--swap-space`
- 健康检查带上 `Authorization: Bearer EMPTY`

涉及文件：

- `src/cegsr/launchers.py`
- `configs/profiles/dual_4090_vllm.yaml`

这两处修复都是“运行兼容性修复”，而不是方法变更。

### 9.3 本轮实际运行命令

#### 第一步：生成服务器脚本

```bash
python scripts/setup_experiment.py --config configs/profiles/dual_4090_vllm.yaml
```

生成结果：

- `outputs/dual_4090/prepare_data.sh`
- `outputs/dual_4090/launch_inference_server.sh`
- `outputs/dual_4090/run_pipeline.sh`
- `outputs/dual_4090/run_ablation.sh`

#### 第二步：准备评测数据

```bash
bash outputs/dual_4090/prepare_data.sh
```

实际数据准备结果：

- `output_path = outputs/data/reasoning_mix_eval.jsonl`
- `num_rows = 300`

但要特别注意，本轮并不是完整 benchmark mix，而是一个“退化后的三数据集子集”：

- `commonsense_qa = 100`
- `ai2_arc = 100`
- `pubmed_qa = 100`

另外两个来源被跳过：

- `gsm8k`: `skipped: ValueError`
- `boolq`: `skipped: HfHubHTTPError`

因此，本轮所有结果都只能解读为：

> 在当前服务器环境下，对 `commonsense_qa + ai2_arc + pubmed_qa` 三数据集、共 300 条样本的结果。

不能直接与完整 mix 的旧结果做严格横向比较。

#### 第三步：启动双卡 vLLM 服务

```bash
bash outputs/dual_4090/launch_inference_server.sh
```

从服务日志可确认：

- `vllm version = 0.18.1`
- 模型路径：`/home/fyk/models/Qwen/Qwen2.5-7B-Instruct`
- `tensor_parallel_size = 2`
- 服务地址：`http://127.0.0.1:8000`
- OpenAI-compatible 路由已成功注册，包括：
  - `/v1/models`
  - `/v1/chat/completions`

这里需要强调一个运行层判断：

> 服务器外网 IP `172.31.162.126` 并不是当前问题的关键。因为推理服务和实验主进程都在同一台服务器本机运行，所以使用 `127.0.0.1:8000` 是正确的。之前出现的 `Connection refused` 并不是 IP 错误，而是因为 vLLM 服务尚未成功启动；而后续出现的 `401` 也不是服务故障，而是鉴权头未带导致的正常拒绝。

#### 第四步：运行完整 pipeline

```bash
bash outputs/dual_4090/run_pipeline.sh
```

主流程耗时：

- `Collect 300`: 约 `15m 35s`
- `Credit 300`: 近乎瞬时
- `Repair 300`: 约 `3m 17s`

输出文件：

- `outputs/dual_4090/raw.jsonl`
- `outputs/dual_4090/annotated.jsonl`
- `outputs/dual_4090/repaired.jsonl`
- `outputs/dual_4090/graph`
- `outputs/dual_4090/training_data`
- `outputs/dual_4090/eval`

#### 第五步：对 raw / annotated 做单独评测

```bash
python scripts/run_eval.py --episodes outputs/dual_4090/raw.jsonl --output-dir outputs/dual_4090/eval_raw
python scripts/run_eval.py --episodes outputs/dual_4090/annotated.jsonl --output-dir outputs/dual_4090/eval_annotated
```

这一组命令非常重要，因为它用同一数据子集回答了：

> 收益到底来自 credit 本身，还是来自 repair？

### 9.4 本轮核心结果

#### 结果 A：raw / annotated / repaired 三段对照

`raw` 结果：

- `accuracy = 0.8033`
- `exact_match = 0.2467`
- `commonsense_qa = 0.82`
- `ai2_arc = 0.85`
- `pubmed_qa = 0.74`

`annotated` 结果：

- `accuracy = 0.8033`
- `exact_match = 0.2467`
- 与 `raw` 完全一致

`repaired` 结果：

- `accuracy = 0.8533`
- `exact_match = 0.2633`
- `repair_coverage = 0.1967`
- `repair_success_rate = 0.2542`
- `num_changed_repairs = 59`
- `commonsense_qa = 0.86`
- `ai2_arc = 0.91`
- `pubmed_qa = 0.79`

#### 结果 B：增益拆解

从同集对照可以直接得到：

- `annotated - raw = 0`
- `repaired - raw = +0.0500 accuracy`
- `repaired - raw = +0.0166 exact_match`

按数据集看：

- `commonsense_qa: 0.82 -> 0.86`，提升 `+0.04`
- `ai2_arc: 0.85 -> 0.91`，提升 `+0.06`
- `pubmed_qa: 0.74 -> 0.79`，提升 `+0.05`

### 9.5 本轮结果说明了什么

#### 结论 1：本轮不需要继续大范围修改方法代码

原因很明确：

1. 双卡服务器上的整条 pipeline 已经跑通；
2. `raw == annotated`，说明 fine-grained credit 这一阶段本身不会直接改变输出结果；
3. `repaired > raw`，而且是稳定的 `+5` 个点 accuracy 提升，说明当前的主要可观测收益仍然来自 selective repair；
4. 这与上一轮日志中形成的主判断完全一致，没有出现推翻式新证据。

因此，当前不值得再进行大范围方法重构。继续大改主代码，收益很可能低于风险。

#### 结论 2：服务器双卡 vLLM 版本已经形成新的“实验运行基线”

这次最重要的工程意义是：

- 不再依赖本地单卡 `hf_local`
- 不再需要每个脚本单独重新加载模型
- 可以把 `7B + vLLM + TP=2` 作为后续 collect / repair / evaluation 的默认运行基线

这对“后续继续进行多次实验”非常关键，因为真正耗时间的不是单次推理质量，而是每轮实验的启动成本和稳定性。

#### 结论 3：当前实验收益仍然主要来自 repair，而不是 graph 或 credit 本身

这次三段对照给出了非常干净的证据：

- `credit` 只是打标签，不直接提升最终正确率；
- 真正改变输出的是 `repair`；
- 因而“verifier-aware fine-grained credit + selective repair”仍然是当前最可信的论文主线；
- graph 在这轮里依然只是被构建出来，还没有被证明是当前稳定收益来源。

#### 结论 4：本轮结果尚不能作为完整论文主表

因为：

- `gsm8k` 缺失
- `boolq` 缺失

所以这轮结果更适合被定义为：

> 双卡服务器运行兼容验证 + 三数据集子集上的方法对照实验

而不是“最终完整 benchmark 主结果”。

### 9.6 这一轮之后，最合理的下一步

#### 推荐立即做的事

1. 保留这套 `dual_4090_vllm` 作为默认服务器运行 profile；
2. 在当前三数据集子集上，如有需要，再跑一次 `run_ablation.sh` 形成统一协议下的同集消融表；
3. 单独排查 `gsm8k` 与 `boolq` 数据准备失败原因，尽快恢复完整 mix。

#### 暂时不推荐做的事

1. 继续大范围重写方法代码；
2. 重新折腾 graph retrieval 主线；
3. 在未恢复完整 benchmark 之前，过早把这次 `0.8533` 写成最终论文主结果。

### 9.7 本轮一句话总结

如果只保留一句总结，这一轮最重要的结论是：

> 双 RTX4090 + vLLM 的服务器运行链路已经稳定跑通；在当前三数据集子集上，`raw == annotated < repaired`，再次证明当前 CEG-SR 的直接收益主要来自 selective repair，而不是 credit 标注本身或 graph retrieval。

## 10. 2026-04-03：七数据集 paper benchmark 恢复、GSM8K 评测修复与全量 ablation

这一轮工作的重点，不再是继续扩展方法模块，而是把“论文级 benchmark 是否真正跑完整、评测协议是否可信、全量 ablation 是否能一次性跑完”这三件事情补齐。

和上一轮相比，这一轮有两个明确目标：

1. 把 benchmark 从“退化后的三数据集子集”恢复为面向论文主实验的七数据集统一评测集；
2. 修复会直接污染实验结论的工程问题，尤其是：
   - `gsm8k` 被 free-form exact match 错误压成 `0.0`；
   - `run_ablation.sh` 长时间无输出，看起来像“卡死”，但实际上只是没有进度可视化。

### 10.1 本轮代码优化

这一轮没有新加方法模块，主要是把评测层与运行层修正确保结果可信。

#### 改动 A：补全七数据集 paper benchmark 数据准备

上一轮已经把数据构建入口改造成可回退的统一 builder；本轮实际在服务器上验证后，已经可以稳定构造如下七个来源组成的 paper benchmark：

- `college_physics`
- `college_chemistry`
- `pubmed_qa`
- `gsm8k`
- `commonsense_qa`
- `ai2_arc`
- `boolq`

涉及文件：

- [`../src/cegsr/data/builders.py`](../src/cegsr/data/builders.py)
- [`../configs/datasets/paper_reasoning_eval.yaml`](../configs/datasets/paper_reasoning_eval.yaml)
- [`../configs/datasets/paper_reasoning_train.yaml`](../configs/datasets/paper_reasoning_train.yaml)
- [`../configs/profiles/paper_benchmark.yaml`](../configs/profiles/paper_benchmark.yaml)
- [`../configs/profiles/dual_4090_vllm_paper.yaml`](../configs/profiles/dual_4090_vllm_paper.yaml)

#### 改动 B：修复 GSM8K 数值评测协议

之前 `gsm8k` 复用通用 `QATask` 的 free-form exact match 逻辑，导致诸如：

- `Final Answer: 42`
- `The answer is 42`
- `42 apples`

这类本质正确的输出，仍然可能因为字符串不完全相等而被记为错误。

本轮在 [`../src/cegsr/tasks/qa.py`](../src/cegsr/tasks/qa.py) 中新增了：

- `numeric_accuracy`
- 对 `gsm8k` / `math_word_problem` 的数值归一化比较
- 对 `Final Answer: ...` / `Answer: ...` / `#### ...` 等数值模式的抽取
- 对带推理过程文本时“最后数值答案”的提取

这一步不是“刷分技巧”，而是把原本错误的评测协议修正为更接近 GSM8K 常见评测方式的协议。

#### 改动 C：补齐 ablation 的进度可见性与健康检查

之前 `run_ablation.sh` 真正的问题不是无法运行，而是：

1. 前三个 baseline：
   - `single_agent`
   - `static_multi_agent`
   - `sirius_lite`
   没有进度条；
2. 生成的 `run_ablation.sh` 没有像 `run_pipeline.sh` 一样先做推理服务健康检查；
3. 因此在服务器上长时间没有输出时，会被误认为“脚本卡死”。

本轮修改后：

- [`../src/cegsr/workflows.py`](../src/cegsr/workflows.py) 为每个方法增加了：
  - `[Ablation] Start ...`
  - `[Ablation] Done ...`
  - baseline 级别的逐样本进度条
- [`../src/cegsr/launchers.py`](../src/cegsr/launchers.py) 为 `run_ablation.sh` 增加了：
  - `/v1/models` 健康检查
  - 与当前 profile 对应的 `launch_inference_server.sh` 提示路径

这一步本质上是“实验可观测性修复”，不是方法改动。

### 10.2 数据准备结果：七数据集 eval 已恢复，train 仍不平衡

#### 10.2.1 训练集构建

运行命令：

```bash
python scripts/prepare_data.py --config configs/datasets/paper_reasoning_train.yaml
```

结果：

- `num_rows = 2010`

分来源如下：

- `college_physics = 5`
- `college_chemistry = 5`
- `pubmed_qa = 400`
- `gsm8k = 400`
- `commonsense_qa = 400`
- `ai2_arc = 400`
- `boolq = 400`

这个结果很重要。它说明：

1. `paper_reasoning_train` 已经可以跑通；
2. 但 `college_physics` / `college_chemistry` 的 train 端目前只回退到了 `dev`，各只有 `5` 条；
3. 因而当前“七数据集完整恢复”首先成立在 **eval benchmark** 层面，而不是训练集规模已经完全平衡。

换句话说，本轮已经解决了“主评测集不完整”的问题，但还没有解决“训练集同样完整且平衡”的问题。

#### 10.2.2 评测集构建

运行命令：

```bash
bash outputs/dual_4090_paper/prepare_data.sh
```

结果：

- `num_rows = 700`
- 七个数据集各 `100` 条

这意味着当前已经具备一个真正可用于主实验的统一七数据集 benchmark：

- `college_physics`
- `college_chemistry`
- `pubmed_qa`
- `gsm8k`
- `commonsense_qa`
- `ai2_arc`
- `boolq`

### 10.3 双 4090 + vLLM 主流程结果

运行命令：

```bash
bash outputs/dual_4090_paper/run_pipeline.sh
```

耗时：

- `Collect 700`: 约 `39m 53s`
- `Credit 700`: 近乎瞬时
- `Repair 700`: 约 `13m 22s`

输出指标：

- `accuracy = 0.8043`
- `exact_match = 0.2429`
- `mcq_accuracy = 0.6857`
- `repair_coverage = 0.2814`
- `repair_success_rate = 0.3046`
- `num_changed_repairs = 197`
- `graph_num_nodes = 2137`
- `graph_num_edges = 50806`

分数据集结果：

- `college_physics = 0.82`
- `college_chemistry = 0.59`
- `pubmed_qa = 0.78`
- `gsm8k = 0.83`
- `commonsense_qa = 0.88`
- `ai2_arc = 0.90`
- `boolq = 0.83`

### 10.4 这一轮最关键的直接结论

#### 结论 1：GSM8K 的 `0.0` 不是模型不会，而是评测协议错了

上一轮最突出的异常是：

- `gsm8k = 0.0`

而本轮只改了评测协议，没有引入新的方法模块后：

- `gsm8k = 0.83`

这说明上一轮 `gsm8k = 0.0` 的主要来源并不是模型完全不会做数学，而是：

- free-form exact match 把本来正确的数值答案系统性误判为错。

因此，这一轮等于修复了一个会直接污染论文主表的评测 bug。

#### 结论 2：七数据集 benchmark 已经可以用于后续主实验

本轮最大的实验价值不只是“又跑了一次”，而是：

- benchmark 不再是三数据集子集；
- `gsm8k` 与 `boolq` 不再缺失；
- `college_physics` 与 `college_chemistry` 也已并入统一评测集；
- 可以开始在完整七数据集上讨论方法增益，而不是继续停留在“子集兼容性验证”。

这意味着项目现在正式进入：

> 在统一七数据集 paper benchmark 上做主实验与消融，而不是继续围绕数据缺失与脚本故障兜圈子。

#### 结论 3：当前 strongest path 仍然是 repair，不是 graph retrieval

虽然这一轮把 graph 也完整构建出来了，但从主流程结果和后面的 ablation 看，当前最稳定的收益来源仍然是 repair 分支，而不是 retrieval 分支。

这一点会在 10.5 的消融结果里表现得非常清楚。

### 10.5 七数据集全量 ablation 结果

运行命令：

```bash
bash outputs/dual_4090_paper/run_ablation.sh
```

这一次脚本不再“看起来卡住”，而是完整显示了各方法的逐个运行过程与结束分数。

核心总表如下：

| method | accuracy | exact_match | repair_coverage | retrieval_proxy |
|---|---:|---:|---:|---:|
| single_agent | 0.2943 | 0.0457 | 0.0 | 0.0 |
| static_multi_agent | 0.7243 | 0.2229 | 0.0 | 0.0 |
| sirius_lite | 0.9543 | 0.2986 | 0.0 | 0.0 |
| ours_wo_graph | 0.8043 | 0.2429 | 0.2814 | 0.0 |
| ours_wo_selective_repair | 0.7171 | 0.2086 | 0.0 | 0.7198 |
| trajectory_level_credit | 0.7229 | 0.2271 | 0.0 | 0.0 |
| repair_only | 0.8043 | 0.2429 | 0.2814 | 0.0 |
| offline_sft_only | 0.7300 | 0.2314 | 0.0 | 0.0 |
| ours_full | 0.7200 | 0.2086 | 0.0 | 0.7228 |

#### 10.5.1 结果解读 A：single-agent 与 static multi-agent 的层级关系清晰

- `single_agent = 0.2943`
- `static_multi_agent = 0.7243`

这说明当前多角色协作本身是有明显收益的；也说明 benchmark 不是“过于简单到随便都高分”。

#### 10.5.2 结果解读 B：repair 依然是当前最稳的正向增益来源

重点看下面几组：

- `static_multi_agent = 0.7243`
- `trajectory_level_credit = 0.7229`
- `offline_sft_only = 0.7300`
- `ours_wo_graph = 0.8043`

这说明：

1. 仅做 trajectory-level credit，并不能直接带来明显提升；
2. 仅做 offline SFT 数据导出，也没有自动转化成在线性能收益；
3. 真正把准确率从 `0.72x` 区间推高到 `0.80x` 区间的，仍然是 repair。

因此，这一轮再次支持：

> 当前 CEG-SR 最可信的直接收益仍然来自 selective repair，而不是 credit 标注本身。

#### 10.5.3 结果解读 C：graph retrieval 依然是负收益

重点看：

- `ours_wo_graph = 0.8043`
- `ours_full = 0.7200`
- `ours_wo_selective_repair = 0.7171`

以及：

- `retrieval_proxy(ours_full) = 0.7228`
- `retrieval_proxy(ours_wo_selective_repair) = 0.7198`

这说明：

1. 检索并不是完全没有命中；
2. 但“命中了”并不等于“最终有帮助”；
3. 当前 retrieval proxy 与最终 answer accuracy 并不对齐；
4. 一旦进入 graph retrieval 路径，结果会从 `0.8043` 明显掉回 `0.72` 左右。

因此，graph retrieval 目前仍不是一个正向主结果来源，甚至在七数据集完整 benchmark 上再次表现为稳定负收益。

#### 10.5.4 结果解读 D：`repair_only` 与 `ours_wo_graph` 当前是同一路径

这两个结果完全一致：

- `repair_only = 0.8043`
- `ours_wo_graph = 0.8043`

这不是偶然，而是当前代码定义下两者本来就是同一条执行路径：

- collect
- credit
- repair
- 不启用 graph retrieval

因此，这个“完全相同的分数”不应被额外过度解读为新科学发现；它主要说明当前 ablation 定义下，这两个名字对应的是同一 artifact。

#### 10.5.5 结果解读 E：`ours_full` 的 `repair_coverage = 0` 不是 repair 没起作用

`ours_full` 指标里有一个很容易误读的地方：

- `repair_coverage = 0.0`
- `num_changed_repairs = 0`

这并不意味着本方法完全没做 repair，而是因为当前 ablation 定义中，`ours_full` 最终评估的是：

- 基于 repaired / annotated graph 构建后的 **fresh retrieved evaluation**

也就是说最终评估文件是 `retrieved_eval.jsonl`，不是 `repaired.jsonl` 本身。因此它不再携带 repair record，`repair_coverage = 0` 在这个 artifact 上是预期现象。

这也说明：

> `run_pipeline.sh` 里的 repaired 主流程结果，不能和 `ours_full` 这个 graph-aware fresh evaluation artifact 直接当成同一个东西横比。

### 10.6 当前最值得重视的新现象：Sirius-lite 异常强

本轮最需要认真对待的不是 graph，而是：

- `sirius_lite = 0.9543`

它不仅高于：

- `static_multi_agent = 0.7243`
- `ours_wo_graph = 0.8043`
- `ours_full = 0.7200`

而且几乎在所有子数据集上都显著领先。

这件事目前至少说明三点：

1. 当前 selective repair 还没有击败“整轨迹失败后全量重写”的强基线；
2. `sirius_lite` 现在应被视为真正的 strongest baseline，而不是弱参考；
3. 后续实验不应该回避这个结果，而应该专门解释：
   - 为什么它这么强；
   - 它带来的增益到底来自：
     - 失败轨迹提示本身，
     - 第二次完整推理预算，
     - 还是整轨迹 regeneration 相比局部 repair 在当前 benchmark 上更合适。

换句话说，这一轮虽然再次确认了 repair 的价值，但也暴露出一个更强、更值得正面分析的 baseline 对手。

### 10.7 本轮综合判断

结合本轮代码修复与全量实验，我认为当前阶段最合理的判断是：

#### 判断 1：完整七数据集 benchmark 已经恢复

这一点已经可以视为完成：

- eval benchmark 七数据集齐全；
- GSM8K 评测不再失真；
- BoolQ 不再缺失；
- 可以开始在统一 benchmark 上稳定重复主实验。

#### 判断 2：当前主增益仍来自 repair，而不是 retrieval

证据非常稳定：

- `ours_wo_graph = 0.8043`
- `ours_full = 0.7200`
- `ours_wo_selective_repair = 0.7171`

graph retrieval 仍然是当前版本的负收益分支。

#### 判断 3：接下来最优先的实验方向，不是继续修 benchmark，而是解释强 baseline

上一阶段的主要问题是：

- benchmark 不完整；
- GSM8K 评测协议错误；
- ablation 长跑不可观测。

这些问题现在都已经被解决或显著缓解。接下来最值得投入时间的，已经不再是“让实验跑起来”，而是：

1. 解释 `sirius_lite` 为什么显著强于 selective repair；
2. 判断这种优势是方法性的，还是协议 / 预算层面的；
3. 进一步设计更公平的对照：
   - double-pass no-feedback
   - full-regeneration with equal budget
   - selective-repair with retrieval-memory support

### 10.8 本轮一句话总结

如果只保留一句总结，本轮最重要的结论是：

> 通过修复 GSM8K 数值评测与 ablation 可观测性，CEG-SR 已经首次在完整七数据集 paper benchmark 上拿到了可信的全量结果；当前 repair 仍然是最稳定的正向收益来源，而 graph retrieval 依旧负收益，同时 Sirius-lite 意外成为新的 strongest baseline，后续实验重点应转向解释并正面对比这一强基线。
