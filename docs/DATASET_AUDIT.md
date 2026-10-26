# 数据集审查

> 文档状态：历史规划材料，已随当前工具和案例一并归档；其中“待完成”表示当时的计划状态，不等于当前代码状态。


## 结论

当前不把 FinQA 和 UCI Taiwanese Bankruptcy 混成一个任务：

- **FinQA**：用于验证可执行 Reward、Rollout、Group 和 RL 诊断链路。
- **Taiwanese Bankruptcy**：用于验证 36 维结构化信用风险分类，更接近客户风控目标。

因此，现阶段不是重新找数据集，而是分别完成工具链基准和风控任务基准。

## FinQA

- 来源：FinQA 官方公开数据。
- 任务：根据财报文本和表格回答数值问题。
- 训练/验证/测试：6,251 / 883 / 1,147 条。
- 监督：标准答案、证据和可执行计算程序；没有自由文本 CoT 真值。
- 适用性：适合验证结果可被程序确定性判分，不能直接代表企业违约预测能力。
- 当前状态：数据转换、程序执行器、Reward、SFT 和受控 GRPO rollout 已实际运行。

## Taiwanese Bankruptcy

- 来源：UCI Taiwanese Bankruptcy Prediction，数据许可记录为 CC BY 4.0。
- 原始数据：6,819 条、95 个特征；当前版本选取 36 个财务特征。
- 当前归档划分：训练 616 条，开发 1023 条，测试 1023 条；随机种子 42；训练集按 1:3 下采样，开发/测试保留自然分布。
- 任务：二分类，标签含义必须在正式报告中明确记录为破产风险/健康企业。
- 监督：只有最终标签，没有可信人工 CoT 或逐步推理真值。
- 适用性：适合做信用风险分类和结果 Reward；不适合直接提供过程 Reward。
- 当前状态：已完成数据准备、基础 Reward、基模/SFT/GRPO/NGRPO 原型评测；指标已归档，但 Precision 仍不足，尚未完成业务验收。

## 必须补齐的审查

1. 固定输入字段、标签字段和 Prompt 模板，确认 Prompt 不包含 Ground Truth。
2. 检查重复样本、训练/测试泄漏和类别比例。
3. 保存数据版本、来源、许可、随机种子和 split 哈希。
4. 使用 Logistic Regression 建立传统基线，重点报告 Recall、Precision、F1/F2 和 False Negative。
5. 不在测试集上调 Prompt、Reward 或训练超参数。

## 当前选择

先用 FinQA 完成 RL 诊断工具的可重复验证，再在 Taiwanese Bankruptcy 上完成风控分类对照。客户真实数据接入后，应重新审查字段语义和时间泄漏，不能直接复用 FinQA Reward。
