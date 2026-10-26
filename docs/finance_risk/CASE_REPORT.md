# 金融风控 RL 实验统一说明

## 1. 一句话结论

本案例观察到了预期现象，但结论需要分成两层：UI/规则发现 GRPO 中同质 group 过多；切换 NGRPO 后 rollout 的有效学习信号明显改善，但仅换算法尚未改善固定测试集结果；继续按前辈实验总结调整训练采样并逐 step 保存、独立评测后，step1 的 Recall、F2 和 FN 才取得实际提升，step2 又迅速发生单类塌缩。

## 2. E1 SFT 到底发生了什么

E1 不是总体准确率为 0。固定测试集有 1023 条，其中 HEALTHY 990 条、BANKRUPTCY 33 条。E1 将 1023 条全部预测为 HEALTHY，因此：

| 指标 | 结果 | 含义 |
|---|---:|---|
| Accuracy | 96.77% | 990 个健康样本被猜对，受类别极不平衡误导 |
| BANKRUPTCY Recall | 0 | 33 个破产样本一个也没识别出来 |
| Precision / F1 / F2 | 0 | 模型没有产生任何有效正类预测 |
| Format Valid Rate | 100% | 输出格式完全可解析，不是格式或 parser 不匹配 |

SFT 使用 616 条训练数据，其中 154 条 BANKRUPTCY、462 条 HEALTHY，比例为 1:3。它的训练目标又主要是学习 `Risk + Evidence` 输出协议。对于只有少量样本、36 维归一化数值且没有专家判定规则的任务，模型最容易降低训练损失的方式是学会固定格式并偏向多数类 HEALTHY，而不是学会稳定的破产判别边界。

因此当前证据支持“类别塌缩”，不支持“因为改变格式导致评测匹配失败”。要进一步确定 SFT 塌缩的主要成因，还需要做平衡 SFT、类别加权 SFT 或更强监督数据的对照实验。

## 3. UI 在 GRPO 中发现了什么

C1 使用 E1 checkpoint、1:3 训练采样、0/1 纯结果 reward 和 GRPO。UI 后端规则从 rollout 中观察到：

- 5 个 step 的平均 Constant Group Rate 为 66.25%，即多数 prompt 的 4 次 rollout 要么全对、要么全错。
- 平均 Effective Group Rate 只有 33.75%，具有组内 Reward 方差的 prompt 较少；这是相对优势信号代理，不是直接读取的反向梯度。
- `finance.constant_group_rate.high` 在 5 个 step 中命中 4 次。
- `finance.positive_groups.no_advantage` 在 5 个 step 中命中 3 次。
- step5 只生成 4 个 BANKRUPTCY、60 个 HEALTHY，策略继续向多数类收缩。

这正是“二分类 + 离散 reward + group size=4”下 GRPO 容易遇到的问题：全对或全错 group 内 reward 方差为 0，GRPO 的相对 advantage 接近 0，模型从这些 group 中几乎学不到方向。

## 4. 换成 NGRPO 后是否好很多

在同一 E1 checkpoint、同一 1:3 数据、同一 seed、同一 0/1 reward 和同样 5 step 下，只将 GRPO 换为 NGRPO：

| Rollout 指标 | GRPO C1 | NGRPO C3 | 变化 |
|---|---:|---:|---:|
| 平均 Constant Group Rate | 66.25% | 48.75% | 降低 17.5 个百分点 |
| 平均 Effective Group Rate | 33.75% | 51.25% | 提高 17.5 个百分点 |
| Constant Group 高风险规则命中 | 4/5 step | 1/5 step | 明显减少 |
| 正类无 advantage 规则命中 | 3/5 step | 0/5 step | 明显减少 |

所以在本次同条件运行中，NGRPO 的 Rollout Reward 方差代理和规则命中优于 GRPO，并与既有算法选择假设一致；由于 Rollout 采样具有随机性且只有单 seed，该结果不能单独证明算法因果优势。

但必须保留后半句：C3 的固定测试集仍然全部预测 HEALTHY，Recall 和 F2 仍为 0。也就是说，“NGRPO 改善了训练信号”已经被观察到，但“只换 NGRPO 就改善最终模型效果”尚未被证明。

## 5. C8 step1 为什么有效，step2 为什么又崩了

后续依据前辈实验总结中“数据配比影响可能大于算法选择”的经验，将训练采样改为 1:1，继续使用 NGRPO 和纯结果 reward，并把 checkpoint 保存频率改为每个 step 一次。

| 阶段 | Accuracy | Precision | Recall | F2 | FN | FP |
|---|---:|---:|---:|---:|---:|---:|
| E1 SFT | 96.77% | 0 | 0 | 0 | 33 | 0 |
| C8 step1 | 50.34% | 4.59% | 72.73% | 18.32% | 9 | 499 |
| C8 step2 | 3.23% | 3.23% | 100% | 14.29% | 0 | 990 |

step1 已从“全部 HEALTHY”移动到能识别 24/33 个破产样本，但代价是 499 个误报。step2 继续朝 BANKRUPTCY 方向移动，最终在确定性测试中把所有样本都判为 BANKRUPTCY，形成相反方向的单类塌缩。

step2 的训练平均 reward 从 0.359375 上升到 0.484375，却不代表固定测试集更好，原因有四个：

1. 训练 reward 只来自当前 step 的 64 条随机 rollout，不是固定的 1023 条测试集。
2. 训练采样是 1:1，而测试集只有 3.23% 正类。全部预测 BANKRUPTCY 在平衡训练 batch 上可接近 50% reward，在自然测试集上 Accuracy 只有 3.23%。
3. rollout 使用随机采样，独立评测使用 temperature=0；采样时看起来仍有两类输出，不代表确定性策略没有塌缩。
4. 0/1 reward 只关心单条是否正确，没有直接约束自然分布下的 Precision、误报数量或类别平衡。

因此训练 reward 用于判断当前 batch 的训练反馈，不能替代固定测试集上的 Precision、Recall、F2 和混淆矩阵。

## 6. 稳定性指标怎么理解

| 指标 | 通俗解释 | 本次 step1 -> step2 | 本次含义 |
|---|---|---:|---|
| Entropy | 模型生成时的“不确定程度”；高表示更分散，低表示更确定 | 0.0145 -> 0.0301 | 不确定性约翻倍，策略正在快速变化；没有跨任务统一安全阈值 |
| KL Loss | 当前策略相对参考策略偏离了多少 | 0.000028 -> 0.039811 | 偏离快速增大，step2 已比 step1 走得远很多 |
| Grad Norm | 本次参数更新力量的大小 | 0.189 -> 0.580 | 更新强度约增至 3 倍，提示策略变化在加速 |
| Clip Fraction | PPO 更新中因变化过大而被裁剪的 token 比例 | 0 -> 0.00488 | 比例仍低，说明此次塌缩不能简单归因于大量 PPO clipping |
| Response Clip | 输出达到最大生成长度的比例 | 1.0 -> 1.0 | 100% 输出都撞到 80-token 上限，是明确异常；标签可解析，但 evidence 经常被截断 |

这些指标不是“越大越好”或“越小越好”。正确看法是联合观察变化速度：本次从 step1 到 step2，KL、Grad Norm、Entropy 同时上升，而独立评测从可用窗口转成单类塌缩，因此 step1 应保留、step2 应拒绝。

## 7. C1-C4 是否是复合 reward

不是。`C1-C4：诊断型 RL -> 单变量优化 -> 独立复评` 描述的是实验流程，不是 reward 公式：

- C1：运行纯结果 GRPO，UI 采集 rollout 与规则命中。
- C2：冻结诊断结论，根据证据选择一个主要优化变量。
- C3：从同一 E1 checkpoint 重新训练，验证该优化变量。
- C4：在固定测试集上独立复评并决定是否验收。

当前主案例一直使用 `credit_risk_outcome_v1`：预测正确为 1，错误为 0。格式和 evidence 只作为诊断字段观察，没有加进训练总 reward。`0.8 outcome + 0.1 format + 0.1 evidence` 仍是后续待单独验证的候选方案，不能与本次算法和采样优化混在一起。

## 8. 当前可以对外汇报的结论

可以表述为：工具在 GRPO rollout 中识别出 Constant Group 过多和正类 group 缺少 advantage，并给出 NGRPO 建议；同条件切换 NGRPO 后，平均 Effective Group Rate 从 33.75% 提升到 51.25%，相关高风险规则命中显著减少。由于固定测试集仍未提升，系统继续依据历史实验中的数据配比结论调整训练采样，并通过逐 step checkpoint 独立评测找到 step1 有效窗口，使 Recall 从 0 提升到 72.73%、F2 从 0 提升到 18.32%、FN 从 33 降到 9。step2 随即发生反向单类塌缩，证明训练 reward 上升不能替代 rollout 诊断和独立评测。

当前不能表述为“NGRPO 已解决业务问题”，因为 step1 Precision 仍只有 4.59%，且尚未完成不同 seed 复现。

## 9. 归档证据

本报告使用的指标、运行清单和 Rollout 汇总已标准化保存于 `results/`。模型权重、完整日志、逐条预测和原始 Rollout 不进入 Git。
