# Rollout 规则说明：Constant Group 与正类无 Advantage

## 1. 统计口径

报告中的 `4/5 step`、`3/5 step` 表示：在 5 个训练 step 中，有多少个 step 满足规则触发条件。它不是样本数量，也不是 Group 数量。

本次实验每个 step 包含 16 个 Group，每个 Group 对同一道 Prompt 生成 4 条 Response。规则先判断各个 Group 的 Rollout 状态，再汇总为 step 级诊断结果。

## 2. Constant Group 比例过高

### 2.1 什么是 Constant Group

当同一个 Group 内 4 条 Response 的正确性和 reward 完全相同时，该 Group 被记为 Constant Group：

```text
全对：reward = [1, 1, 1, 1]
全错：reward = [0, 0, 0, 0]
```

计算方式为：

```text
Constant Group Rate =（全对 Group 数 + 全错 Group 数）/ 总 Group 数
```

本案例的规则阈值为 `0.5`。当一个 step 的 Constant Group Rate 大于或等于 50% 时，触发 `finance.constant_group_rate.high`。

### 2.2 为什么它是风险信号

标准 GRPO 依靠同一 Prompt 的多条 Response 之间的 reward 差异计算相对 advantage。对于全对或全错 Group，组内 reward 方差为 0，归一化后的相对 advantage 通常为 0 或接近 0：

```text
reward    = [1, 1, 1, 1]
advantage = [0, 0, 0, 0]

reward    = [0, 0, 0, 0]
advantage = [0, 0, 0, 0]
```

而混合 Group 可以提供组内比较方向：

```text
reward = [1, 1, 0, 0]
```

因此，Constant Group 比例过高表示当前 step 中可供 GRPO 比较和学习的 Group 较少。该规则是学习信号的代理诊断，不等于直接测量实际反向梯度，也不表示模型有 50% 的预测错误。

### 2.3 GRPO 的 4/5 step 如何得到

| Step | 全对 Group | 全错 Group | Constant Group | 比例 | 是否命中 |
|---|---:|---:|---:|---:|---|
| 1 | 8 | 2 | 10/16 | 62.5% | 是 |
| 2 | - | - | 5/16 | 31.25% | 否 |
| 3 | 9 | 5 | 14/16 | 87.5% | 是 |
| 4 | 8 | 4 | 12/16 | 75% | 是 |
| 5 | 9 | 3 | 12/16 | 75% | 是 |

所以 GRPO 实验中该规则命中 `4/5 step`。NGRPO 实验中只有 step 5 达到阈值，因此命中 `1/5 step`。

## 3. 正类无 Advantage

### 3.1 什么是正类

本次金融风控任务中的正类是 `BANKRUPTCY`，即需要重点识别的高风险企业样本。

### 3.2 什么是正类无 Advantage

如果正类 Prompt 的 4 条 Response 全部答对或全部答错，那么该 Group 内没有 reward 差异，标准 GRPO 无法通过组内比较判断哪条 Response 更好。

最需要关注的是正类全错：

```text
真实标签：BANKRUPTCY
reward：  [0, 0, 0, 0]
```

模型没有生成正确答案，但 4 条 Response 又同样错误，标准 GRPO 缺少相对排序依据。这个现象被记为正类缺少 advantage 信号。

### 3.3 GRPO 的 3/5 step 如何得到

- Step 1：正类中有 1 个全对 Group、2 个全错 Group，规则命中。
- Step 3：正类中有 4 个全错 Group，规则命中。
- Step 4：正类中有 4 个全错 Group，规则命中。
- Step 2、Step 5：未满足该历史规则的严格触发条件。

所以 GRPO 实验中命中 `3/5 step`，NGRPO 实验中命中 `0/5 step`。

Step 5 虽然出现了 `near_all_wrong` 正类 Group，但历史规则只把严格的全对、全错 Constant Group 作为命中条件，因此该 step 没有被计入。这是规则边界，不代表 Step 5 的正类学习状态良好。

## 4. 为什么 NGRPO 的命中次数减少

本案例中的 NGRPO 在计算归一化参考时加入虚拟最大 reward 样本。对于低分或全错 Group，真实 Response 可以相对于虚拟高分参考获得非零方向，从而缓解标准 GRPO 在全错 Group 中没有相对信号的问题。

实验中观察到：

| 规则 | GRPO | NGRPO | 观察结果 |
|---|---:|---:|---|
| Constant Group 比例过高 | 4/5 step | 1/5 step | 明显减少 |
| 正类无 Advantage | 3/5 step | 0/5 step | 明显减少 |

该结果说明 NGRPO 实验中的 Rollout 学习信号代理优于 GRPO，但不能单独证明每个 Group 的实际梯度，也不能替代固定测试集评测。

## 5. 推荐展示名称

“高风险规则命中”容易被理解为识别到了高风险企业。前端和汇报材料建议使用以下名称：

- `同质 Group 比例过高`
- `高风险正类缺少组内学习信号`

规则说明应同时展示触发阈值、实际数值和触发原因，例如：

```text
同质 Group 比例过高
实际值：12/16（75%）
阈值：>= 50%
原因：该 step 中 8 个 Group 全对、4 个 Group 全错，只有 4 个 Group 存在组内 reward 差异。
```

## 6. 结论边界

这些规则能够说明 Rollout 是否提供了足够的组内相对学习信号，但不能直接回答：

- 某个 Group 最终贡献了多少实际梯度；
- 模型参数一定向正确业务方向更新；
- NGRPO 一定优于 GRPO；
- 训练 reward 改善后，固定测试集指标一定改善。

因此规则结果需要与独立测试集的 Precision、Recall、F2、混淆矩阵，以及 KL、Grad Norm 等训练稳定性指标联合解释。
