# Reward 设计与验收

> 文档状态：历史规划材料，已随当前工具和案例一并归档；其中“待完成”表示当时的计划状态，不等于当前代码状态。


## 原则

Reward 是训练信号，不是正确率的保证。增加 Reward 分量可能只增加解释信息，也可能改变 Group 内排序；只有独立评测证明指标提升，才能认为模型效果提升。

## 第一阶段：纯结果 Reward

Taiwanese Bankruptcy 首轮 GRPO/NGRPO 只使用结果 Reward：

```text
合法标签且预测正确：1
合法标签但预测错误：0
无法解析合法标签：单独记录为 invalid，不把格式奖励混入结果奖励
```

训练对照阶段必须保持这个定义不变，避免算法差异和 Reward 差异混在一起。

类别不平衡不直接通过修改 Group Reward 解决。独立评测必须报告正类 Recall、Precision、F1/F2、False Negative 和混淆矩阵；是否需要类别敏感 Reward，要等纯结果实验后再用对照实验确认。

## 第二阶段：复合 Reward

只有纯结果 Reward 的链路稳定后，才单独增加复合 Reward。候选版本：

```text
R_total = 0.8 * R_outcome + 0.1 * R_format + 0.1 * R_evidence
```

该公式只是待验证配置，不是默认最佳方案。每次只增加一个可解释变量，并保留 Reward 版本号。

### Reward 分量定义

- `outcome`：预测标签是否正确。
- `format`：输出是否符合协议，不能把所有格式合法输出都当成任务正确。
- `evidence`：引用的字段是否真实存在于输入；事实引用正确不代表风险判断正确。
- `process`：当前暂缓，除非存在明确且可验证的中间步骤标准。

## 当前实现审查

- FinQA 已有确定性执行 Reward，范围 `-4` 到 `+4`，可区分程序、答案和证据错误；用于 RL 诊断基准。
- 风控 Reward 已有纯结果 Reward、类别加权 Reward、独立单元测试和正式原型训练接入；复合 Reward 仍未作为默认训练规则。
- 当前不能把诊断阶段重新计算的分数称为训练时 Reward；必须保留实际训练返回值及其来源。

## 必须新增的 Reward 验收

1. 对正确/错误/非法格式/缺失证据/证据正确但标签错误逐项写单测。
2. 在不启动模型的情况下，用固定样例检查纯结果 Reward 的输出。
3. 单独运行复合 Reward，对比各分量和总分。
4. 对每个 Prompt Group 计算总 Reward 和各分量的 unique count、range、std、informative pair ratio。
5. 检查分量是否饱和：如果 Group 内全部相同，该分量不提供相对学习信号。
6. 将训练 Reward、Reward 分量、KL penalty 和独立 Verifier 结果分开保存。

## UI 需要展示

每个实验和 Group 必须能看到：Reward 规则 ID/版本、总分、分量原始值、权重、加权贡献、评分状态、独立评测结果及原始 Response。旧日志没有分量时显示“未提供”，不能伪装成 0。
