# RL 训练诊断开发流程总结

## 目标

快速回答三个训练问题：rollout 是否有效、组内 reward 是否提供学习信号、训练指标是否同步出现异常。训练期不接入 AI 评审，不推断经验难度，不自动给出算法迁移结论。

## 已走通的流程

1. 修复并核验数学数据输入，确认 prompt、标准答案和 reward 链路真实生效。
2. 给 VERL rollout 输出补充稳定的 step、group、rollout、reward、`acc` 和 `pred` 证据。
3. 跑通 GRPO 小规模训练并保存逐 step rollout 与训练日志。
4. 实现 NGRPO 运行时补丁，可在现有镜像中应用、检查和恢复。
5. 实现不依赖新增 Python 包的解析、组内信号统计、异常样本集中导出和 Jupyter 监控。

## 当前最小闭环

训练启动前检查输入与评分器；训练中按 step 查看 reward、有效输出、组内全同/混合情况及 KL、entropy、gradient norm；训练后集中复盘异常 prompt、回答、reward、预测值和验证结果。所有实验产物保留在运行目录，不进入 Git。

## 当前事实边界

- `difficulty` 只接受数据集原始字段或显式映射，缺失即跳过。
- `acc` 存在时可报告全对、全错、接近全对、接近全错和混合组。
- `acc` 不存在时只报告正/非正 reward 比例，不能将其直接称为正确率。
- 阈值只负责筛选值得复盘的组，不解释问题成因。
- 梯度抖动只能与 rollout 和 reward 变化做时间关联，不能仅凭图表断言因果。

## 下一步顺序

1. 用真实 GRPO 运行验证统一异常样本文件和 Jupyter 展示。
2. 固定同一模型、数据、seed 和训练预算，跑 GRPO 与 NGRPO 对照。
3. 比较有效组比例、异常组比例、reward、KL、entropy、gradient norm 和最终评测。
4. 只有证据显示 GRPO 的组内信号长期不足时，才讨论 NGRPO 是否改善该问题。
