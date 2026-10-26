# 金融风控 RL 诊断优化案例：PPT 汇报材料

## 汇报主线

本案例不把“训练成功”定义为任务跑通，而是展示一条可复盘的优化链路：

> SFT 学会输出格式但发生多数类塌缩 -> GRPO 训练效果不佳 -> 前端 UI 从 rollout 中发现全对/全错 Group 过多、有效学习信号不足 -> 依据历史实验沉淀的算法选择经验切换 NGRPO -> NGRPO 改善 Group 学习信号 -> 继续调整训练采样并逐 Step 保存 checkpoint -> Step1 在独立测试集上取得阶段性提升 -> Step2 再次发生单类塌缩，UI 稳定性趋势与独立评测共同解释退化原因。

这条主线体现的不是“NGRPO 一定优于 GRPO”，而是工具如何帮助研发人员发现问题、选择优化方向、捕捉短暂有效窗口，并阻止继续使用已经退化的 checkpoint。

---

## 第 1 页：案例目标

### 标题

金融风控强化学习诊断与优化案例

### 页面内容

- 任务：根据企业 36 维财务指标判断 `BANKRUPTCY` 或 `HEALTHY`。
- 数据：UCI Taiwanese Bankruptcy Prediction。
- 模型起点：同一个金融风控 SFT checkpoint。
- 诊断工具：实时解析训练日志和 rollout，恢复 Prompt Group，识别全对、全错、混合 Group，并展示 Reward、KL、Entropy、梯度和截断趋势。
- 业务指标：重点关注破产类 Recall、Precision、F2、False Negative，而不是只看 Accuracy。

### 建议图示

使用一条流程箭头：

`SFT -> GRPO -> UI 诊断 -> NGRPO/采样优化 -> checkpoint 评测 -> 最佳模型选择`

### 讲解口径

本案例的重点不是比较一次训练的最终分数，而是验证工具能否在训练过程中发现问题，并把诊断结论转化为下一轮可验证的优化动作。

---

## 第 2 页：SFT 学会格式，但没有学会破产识别

### 标题

SFT 基线：格式正确，分类能力发生塌缩

### 核心数据

| 指标 | E1 SFT |
|---|---:|
| 测试样本 | 1023 |
| HEALTHY / BANKRUPTCY | 990 / 33 |
| Format Valid Rate | 100% |
| Accuracy | 96.77% |
| BANKRUPTCY Recall | 0% |
| F2 | 0 |
| False Negative | 33 |

### 核心结论

- 模型能够稳定输出 `Risk + Evidence` 格式，排除 parser 或输出协议不匹配。
- 模型将全部样本预测为 `HEALTHY`。
- 96.77% Accuracy 来自测试集类别不平衡，不代表模型具备风险识别能力。
- SFT 训练集为 154 个破产样本、462 个健康样本，比例为 1:3；模型学会了格式，同时偏向多数类。

### 建议图表

混淆矩阵：

| | 预测 BANKRUPTCY | 预测 HEALTHY |
|---|---:|---:|
| 实际 BANKRUPTCY | 0 | 33 |
| 实际 HEALTHY | 0 | 990 |

### 讲解口径

SFT 并非“完全没有准确率”，而是总体 Accuracy 很高、正类 Recall 为零。该现象说明金融风控任务不能只使用 Accuracy 验收，必须拆分正类指标和混淆矩阵。

---

## 第 3 页：GRPO 没有解决问题

### 标题

GRPO Rollout 中大量 Group 缺少有效学习信号

### 实验条件

- 起点：E1 SFT checkpoint。
- 算法：GRPO。
- 训练采样：BANKRUPTCY : HEALTHY = 1:3。
- Reward：纯结果 0/1 Reward。
- Group Size：每个 Prompt 生成 4 个 rollout。
- 训练长度：5 Step。

### UI 观察结果

| Rollout 指标 | GRPO C1 |
|---|---:|
| 平均 Constant Group Rate | 66.25% |
| 平均 Effective Group Rate | 33.75% |
| Constant Group 高风险规则命中 | 4 / 5 Step |
| 正类无 Advantage 规则命中 | 3 / 5 Step |
| Step5 输出 BANKRUPTCY / HEALTHY | 4 / 60 |

### 现象解释

- 全对或全错 Group 内的 4 个 rollout Reward 完全相同。
- GRPO 依赖组内 Reward 差异计算相对 Advantage。
- Reward 方差为零时，该 Group 几乎不能提供更新方向。
- 模型继续向多数类 `HEALTHY` 收缩。

### 建议图表

堆叠柱状图：每个 Step 展示全对、全错、近全对、近全错和混合 Group 数量。

突出 Step1、Step3、Step4、Step5 的 Constant Group 高风险规则命中。

### 讲解口径

前端 UI 并不是只展示 Reward 曲线，而是恢复同一 Prompt 的多次 rollout。正是 Group 视角让我们看到：GRPO 的主要问题不是 Reward 数值低，而是大量 Group 没有相对学习信号。

---

## 第 4 页：根据沉淀经验选择 NGRPO

### 标题

诊断结论驱动算法调整

### 决策依据

历史金融分类实验沉淀的算法选择经验指出：

- 二分类任务使用离散 Reward 时，同质 Group 概率较高。
- 小 Group 场景下，GRPO 容易因 Group 内方差为零而失去学习信号。
- 本仓库 NGRPO 实现会在组统计量中加入虚拟最大 Reward 样本，使同分但低于最大值的 Group 获得非零负 Advantage；全组均已达到最大 Reward 时仍然没有组内信号。
- DAPO 会过滤全对、全错 Group；当前正类稀少且 Group Size 只有 4，过滤可能进一步减少有效样本，因此暂不选择 DAPO。

### 优化动作

第一步只改变算法：

`GRPO -> NGRPO`

保持 E1 checkpoint、1:3 数据、seed、Reward、Group Size 和训练 Step 不变，观察 rollout 学习信号是否改善。

### 建议图示

左侧显示 GRPO 同质 Group：`[0, 0, 0, 0] -> Advantage ≈ 0`。

右侧显示 NGRPO：`[0, 0, 0, 0, virtual max] -> 非零 Advantage`。

### Skill 规则映射

| 当前现象 | 历史经验规则 | 建议动作 | 本次用途 |
|---|---|---|---|
| 二分类、0/1 离散 Reward、同质 Group 多 | GRPO 在零方差 Group 上 Advantage 退化 | GRPO 改为 NGRPO | 改善同质 Group 的学习信号 |
| NGRPO 后仍偏向单一类别 | 数据配比的影响可能大于算法选择 | 保持 NGRPO，调整训练采样比例 | 增加正类探索机会 |
| 1:3 容易全负、1:1 容易全正 | 极端配比会把策略推向不同单类 | 用 1:1 做短程压力实验并逐 Step 保存；最终配置继续验证 1:2 等折中比例 | 捕捉有效窗口，同时防止过度训练 |
| KL、Entropy、Grad Norm 同时快速上升 | 策略正在加速偏离 | 缩短训练、降低学习率、增强 KL 或回退 checkpoint | Step2 退化后拒绝该 checkpoint |

### 讲解口径

这里所说的 Skill 是历史实验沉淀的算法选择经验，目前来源于《RL 算法选择指南》和实验总结，尚未封装成自动执行的 Skill 包。它提供的是可验证假设，不是跨数据集和模型必然成立的固定答案。

---

## 第 5 页：NGRPO 改善了 Rollout 学习信号

### 标题

同条件切换 NGRPO 后，低信息 Group 明显减少

### 对比数据

| Rollout 指标 | GRPO C1 | NGRPO C3 | 变化 |
|---|---:|---:|---:|
| 平均 Constant Group Rate | 66.25% | 48.75% | -17.5 个百分点 |
| 平均 Effective Group Rate | 33.75% | 51.25% | +17.5 个百分点 |
| Constant Group 规则命中 | 4 / 5 | 1 / 5 | 明显减少 |
| 正类无 Advantage 规则命中 | 3 / 5 | 0 / 5 | 不再命中 |

### 核心结论

- 本次 NGRPO 运行观察到 rollout 组内信号代理改善。
- UI 中高风险规则命中明显减少。
- 该结果与算法选择假设一致，但单次随机采样对比不能单独证明算法因果效果。
- 但同条件 C3 的固定测试集仍全部预测 `HEALTHY`，只换算法尚未改善业务指标。

### 建议图表

并列柱状图：

- Constant Group Rate：66.25% vs 48.75%。
- Effective Group Rate：33.75% vs 51.25%。
- 规则命中次数：4 vs 1、3 vs 0。

原始图表数据：`grpo_ngrpo_rollout_comparison.csv`。

### 讲解口径

这一步不能汇报成“NGRPO 已经解决业务问题”。准确说法是：NGRPO 解决了部分训练信号退化问题，但模型最终效果还受到数据配比和训练推进速度影响。

---

## 第 6 页：继续优化并找到 Step1 有效窗口

### 标题

NGRPO + 平衡训练采样 + 逐 Step 评测找到有效 Checkpoint

### 第二阶段优化

根据历史经验中“数据配比影响可能大于算法选择”的结论，本轮将 1:1 作为短程压力实验，用于主动扩大正类探索窗口；它不是最终推荐配比，历史经验同时预警 1:1 可能走向全正类：

- 训练采样由 1:3 调整为 1:1，并限定为逐 Step 观察。
- 固定自然分布测试集不变，避免修改测试标准。
- 继续使用 NGRPO 和纯结果 Reward。
- 每个 Step 保存 checkpoint，并立即进行固定测试集独立评测。

### Step1 结果

| 指标 | E1 SFT | C8 Step1 |
|---|---:|---:|
| Recall | 0% | 72.73% |
| Precision | 0% | 4.59% |
| F2 | 0 | 18.32% |
| False Negative | 33 | 9 |
| True Positive | 0 | 24 |

### 核心结论

- Step1 将破产样本识别数量从 0 提升到 24。
- FN 从 33 降到 9，F2 出现实际提升。
- 但产生 499 个 False Positive，Precision 仍很低。
- Step1 是“有效探索窗口”，不是可直接上线的业务模型。

### 建议图表

分组柱状图：E1 与 Step1 的 Recall、F2、FN。

同时展示 Step1 混淆矩阵，明确收益与代价。

### 讲解口径

Step1 的效果来自组合动作，不能全部归因于 NGRPO：NGRPO 改善了学习信号，1:1 采样增强了正类探索，逐 Step 评测帮助我们及时保留了短暂有效 checkpoint。

---

## 第 7 页：Step2 Reward 更高，但模型反而崩溃

### 标题

训练 Reward 上升不等于固定测试集效果上升

### Step1 与 Step2 对比

| 指标 | Step1 | Step2 |
|---|---:|---:|
| 训练平均 Reward | 0.3594 | 0.4844 |
| Rollout Effective Group Rate | 62.5% | 75.0% |
| Rollout BANKRUPTCY / HEALTHY | 21 / 43 | 23 / 41 |
| KL Loss | 0.000028 | 0.039811 |
| Entropy | 0.0145 | 0.0301 |
| Grad Norm | 0.189 | 0.580 |
| 独立测试 Recall | 72.73% | 100% |
| 独立测试 Precision | 4.59% | 3.23% |
| 独立测试 F2 | 18.32% | 14.29% |
| False Positive | 499 | 990 |

### UI 可以看到的风险趋势

- KL Loss 快速上升，模型明显偏离参考策略。
- Grad Norm 增长约 3 倍，参数更新强度迅速增大。
- Entropy 约翻倍，策略分布正在快速变化。
- Response Clip 始终为 100%，所有输出达到 80-token 上限。
- Rollout 中 BANKRUPTCY 输出继续增加，但 sampled rollout 尚未直接表现为全部正类。

### 独立评测确认的问题

Step2 在 temperature=0 的固定测试集上将全部 1023 条样本预测为 `BANKRUPTCY`：

- TP = 33。
- FP = 990。
- Accuracy = 3.23%。
- F2 从 18.32% 降至 14.29%。

### 建议图表

使用双区域展示：

1. 左侧折线：Reward、KL Loss、Grad Norm 从 Step1 到 Step2 的变化。
2. 右侧混淆矩阵：Step1 与 Step2 对比。

### 讲解口径

UI 的 rollout 与稳定性指标能够发现“策略正在快速偏移”的风险，但仅凭 sampled rollout 不能断言确定性模型已经单类塌缩。独立测试集评测负责确认退化，UI 负责解释退化发生前后的训练趋势。两者必须结合。

---

## 第 8 页：工具价值与案例结论

### 标题

从训练监控升级为“诊断—优化—验证”闭环

### 本案例验证的工具能力

1. 从原始 rollout 恢复 Prompt Group，而不是只看平均 Reward。
2. 自动识别全对、全错、近全对、近全错和混合 Group。
3. 通过规则发现 GRPO 中 Constant Group 过多和正类缺少 Advantage。
4. 使用历史实验经验选择 NGRPO，并验证 rollout 学习信号改善。
5. 将训练 Reward、稳定性趋势和独立评测分开解释。
6. 通过逐 Step checkpoint 评测保留 Step1，拒绝 Reward 更高但已经塌缩的 Step2。
7. 保存原始日志、rollout、规则命中、指标和报告，支持后续复盘及 PPT 汇报。

### 最终结论

- SFT 学会了格式，但没有学会有效的破产识别。
- GRPO 中大量全对/全错 Group 导致有效学习信号不足。
- 本次 NGRPO 运行改善了 rollout Group 信号代理，但不能单独保证最终业务效果。
- NGRPO、训练采样调整和逐 Step 评测共同找到 Step1 有效窗口。
- Step2 表明训练 Reward 上升可能伴随策略塌缩，必须结合 UI 趋势和独立评测选择 checkpoint。

### 限制

- Step1 Precision 只有 4.59%，误报仍不可接受。
- 所有 rollout 输出均达到长度上限，需要修复 Response Clip 问题。
- 当前仅完成 seed=42，尚未证明可重复性。
- 当前案例证明了诊断工具价值，尚未完成业务模型验收。

---

## 第 9 页：下一步计划

### 实验改进

- 使用 seed 43、44 复现 Step1 有效窗口。
- 预先声明最低 Precision 和 Recall 验收门槛。
- 测试 1:2 采样、降低学习率和更严格 KL 约束，缓解 Step2 过度偏移。
- 缩短 Evidence 输出或提高最大响应长度，消除 100% Response Clip。
- 在纯结果 Reward 基线稳定后，再单独验证类别加权 Reward 或复合 Reward。

### 前端改进

- 在规则命中后显示“推荐 NGRPO”及其证据和适用条件。
- 将 GRPO、NGRPO 和后续优化实验显示为父子实验链路。
- 在前端接入独立测试集 Precision、Recall、F2和混淆矩阵。
- 增加 checkpoint 时间线，并标记“候选、最佳、退化、拒绝”。
- 支持一键导出本案例的图表、指标表和证据索引。

---

## 汇报时需要避免的表述

不要说：

- “SFT 准确率为 0。”
- “只要换成 NGRPO，模型效果就好了。”
- “Step2 Reward 更高，所以 Step2 更好。”
- “UI 仅凭 rollout 已经准确预测到 Step2 会全部输出 BANKRUPTCY。”

建议说：

- “SFT 总体 Accuracy 为 96.77%，但破产类 Recall 为 0，发生多数类塌缩。”
- “本次 NGRPO 运行的 Effective Group Rate 从 33.75% 提升到 51.25%，表明组内 Reward 方差代理改善。”
- “最终 Step1 提升来自 NGRPO、训练采样调整和逐 Step checkpoint 评测的组合。”
- “UI 发现策略快速偏移风险，固定测试集进一步确认 Step2 已发生单类塌缩。”

## 材料索引

- 统一案例说明：`UNIFIED_CASE_REPORT.md`
- PPT 图表数据：`grpo_ngrpo_rollout_comparison.csv`
- 指标对比数据：`metrics_comparison.csv`
- C8 Rollout 数据：`rollout_steps.csv`
- C8 稳定性数据：`training_stability.csv`
- 完整结构化证据：`case_evidence.json`
