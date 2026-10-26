# 实验矩阵与下一步计划

> 文档状态：历史规划材料，已随当前工具和案例一并归档；其中“待完成”表示当时的计划状态，不等于当前代码状态。


## 实验路线

| ID | 初始化 | 数据 | 训练/评测 | 目的 | 状态 |
|---|---|---|---|---|---|
| E0 | Qwen 8B 基模 | Taiwanese Bankruptcy | 独立评测 | 建立基模风控基线 | 已完成 |
| E1 | Qwen 8B 基模 | Taiwanese Bankruptcy | 格式 SFT 后独立评测 | 判断 SFT 是否只改善格式 | 已完成：格式正确但类别塌缩 |
| E2 | E1 同一 Checkpoint | Taiwanese Bankruptcy | 纯结果 GRPO | 建立 RL 基线并采集 rollout 诊断 | 已完成短程诊断：同质组多、输出截断 |
| E3 | E1 同一 Checkpoint | Taiwanese Bankruptcy | 纯结果 NGRPO | 与 GRPO 公平对照 | 已完成原型探索，待多 seed 公平复现 |
| R1 | 不启动模型 | 固定样例 | 纯结果 Reward | 验证评分逻辑 | 已完成 |
| C1-C4 | E1 同一 Checkpoint | Taiwanese Bankruptcy | 诊断型 RL → 单变量优化 → 独立复评 | 形成可汇报的“发现问题—优化—效果提升”案例 | 原型链路完成，待 precision 门槛与多 seed 验收 |
| R2 | 不启动模型 | 固定样例 | 复合 Reward | 验证分量和组内区分度 | R1 后 |
| A1-A7 | 不同受控配置 | 合成/隔离数据 | 异常注入 | 验证诊断规则 | 后续 |

## E0/E1 对照要求

- 同一模型 tokenizer、Prompt、测试集和解码参数。
- 不向 Prompt 注入标签或任何测试集答案。
- 保存原始 Response、解析结果、混淆矩阵和错误样例。
- 重点比较正类 Recall、Precision、F1/F2、False Negative；Accuracy 只作为辅助。
- 记录格式合法率和平均响应长度，区分格式变化与预测变化。

## E2/E3 公平性要求

E2 和 E3 必须使用：

- 同一个 E1 Checkpoint。
- 同一数据划分、纯结果 Reward 版本、随机种子集合和训练预算。
- 相同 rollout 数量、温度、最大响应长度、学习率和批量配置。
- 同一独立测试集和同一评测脚本。

先检查 NGRPO 实际实现的 Advantage、过滤或采样差异，不根据算法名称推断效果。单次短实验只能作为链路检查，不能据此断言 NGRPO 优于 GRPO。

## 每次实验必须保存

- 模型 Checkpoint 来源和版本。
- 数据版本、split 哈希和随机种子。
- Reward 规则 ID、代码版本和完整配置。
- 训练日志、Rollout JSONL、Group 统计和 UI 快照。
- 独立评测结果、错误样例和计算成本。

## 诊断验收指标

### Rollout/Group

- 格式合法率、解析失败率、截断率。
- Constant、Near-constant、Effective Group 比例。
- Unique Reward Count、Reward range/std、Informative Pair Ratio。
- 各 Reward 分量的组内区分度。

### 训练稳定性

- Reward、Advantage、Entropy、KL、Clip Fraction、Grad Norm。
- Step 耗时、生成耗时、Token Throughput。
- 训练 Reward 与独立 Verifier 的一致率。

### 异常实验

先用合成数据验证常数 Reward、截断、评分器异常、温度变化和异常指标序列；再用隔离的真实训练观察现象。每个异常实验都要记录假设、改动、对照、期望、实际、证据和 PASS/FAIL。

## 执行顺序

1. 完成 E0/E1：同一测试集的基模/SFT 独立评测。
2. 检查 Reward Manager 是否支持携带额外评分信息；没有接口前不修改训练主链路。
3. 用 E1 做短程 E2 GRPO，保留原始 rollout、Group 统计和规则证据。
4. 根据 E2 诊断只选一个优化变量，执行 C2 决策并记录依据。
5. 从同一 E1 做 C3 优化型 GRPO，比较固定测试集指标和诊断证据。
6. 若 C3 有提升，再做不同 seed 复现并沉淀案例报告；否则记录为失败案例。
7. 在案例结论稳定后，再从同一 E1 做 E3 NGRPO 对照，最后执行 R2 复合 Reward 和异常注入。

## 当前明确不做

- 不启动长时间 GRPO/NGRPO。
- 不强制加入 CoT 或未经验证的教师模型推理文本。
- 不用复合 Reward 直接替代纯结果 Reward。
- 不把 FinQA 的程序 Reward 直接迁移到信用风险分类。
- 不用一次短实验宣称某个 RL 算法更好。
