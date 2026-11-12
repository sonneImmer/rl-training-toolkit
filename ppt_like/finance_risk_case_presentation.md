> 归档说明（2026-11-12）：本文件保留历史汇报/经验草稿。通用阈值、因果判断和优化建议尚未全部校准，不作为已验证的自动诊断规则；当前状态与限制以`docs/CURRENT_STATUS_20261112.md`及案例指标为准。

# 金融风控 RL 案例：RL Training Toolkit 如何介入训练流程

## 一、案例一句话

> RL Training Toolkit 不是简单展示训练曲线，而是在训练过程中发现问题、指导下一轮实验、验证优化效果，并帮助研发人员保留有效 checkpoint、拒绝退化 checkpoint。

本案例的真实主线不是“训练指标一直上升”，而是：

```text
SFT 基线
  -> 发现模型只会输出格式，不会识别破产
  -> GRPO 训练
  -> 工具发现大量 Group 没有有效学习信号
  -> 切换 NGRPO
  -> 有效学习信号改善
  -> 调整训练采样比例
  -> 逐 Step 评测 checkpoint
  -> 找到 Step1 有效窗口
  -> 发现 Step2 继续训练后发生策略塌缩
  -> 保留 Step1，拒绝 Step2
```

---

## 二、这个案例在做什么？

这是一个金融风控二分类任务。

模型需要根据企业的 **36 个财务指标**，判断企业属于：

- `BANKRUPTCY`：可能破产
- `HEALTHY`：健康

固定测试集共有 1023 条样本：

| 类别 | 数量 |
|---|---:|
| HEALTHY | 990 |
| BANKRUPTCY | 33 |

数据存在明显的类别不平衡。因此，如果模型把全部样本都预测为 `HEALTHY`，Accuracy 也可以达到：

```text
990 / 1023 = 96.77%
```

但它实际上一个破产企业都没有识别出来。

因此本案例重点不看 Accuracy，而是关注：

- **Recall**：破产样本识别出来多少；
- **Precision**：预测为破产的样本中，有多少是真的；
- **F2**：更重视漏掉破产样本的综合指标；
- **False Negative**：漏掉了多少破产企业；
- **False Positive**：误报了多少健康企业。

---

## 三、RL Training Toolkit 在案例中的介入位置

```text
训练日志和 Rollout
        |
        v
工具解析训练过程和模型输出
        |
        v
恢复 Prompt Group，分析 Reward 和 Advantage
        |
        v
发现学习信号不足、策略偏移或输出退化
        |
        v
调整算法、采样比例或训练强度
        |
        v
逐 Step 保存 checkpoint 并进行独立测试
        |
        v
保留有效模型，拒绝退化模型
```

工具介入了四个关键环节：

1. **看见训练发生了什么**：读取训练日志和 Rollout。
2. **解释模型为什么没有学好**：分析 Prompt Group 和学习信号。
3. **支持下一轮优化**：根据证据调整 GRPO/NGRPO、采样比例等。
4. **判断哪个 checkpoint 值得保留**：结合独立测试和稳定性指标做决策。

---

# 四、建议的 PPT 讲解顺序

## 第 1 页：案例任务和验收标准

### 建议标题

**金融风控 RL 案例：从训练监控到 checkpoint 选择**

### 页面内容

```text
任务：根据 36 个财务指标识别 BANKRUPTCY / HEALTHY
数据：1023 条固定测试样本，其中破产样本 33 条
重点：不能只看 Accuracy，需要关注 Recall、F2 和漏报
```

### 讲解方式

> 这个实验不是普通的二分类准确率展示。因为健康样本占多数，模型即使全部预测为健康，也能得到很高的 Accuracy。对于金融风控来说，真正重要的是不要漏掉破产企业，所以我们重点关注破产 Recall、F2 和 False Negative。

---

## 第 2 页：SFT 基线问题

### 建议标题

**SFT 学会了输出格式，但没有学会识别破产**

### 关键结果

| 指标 | E1 SFT |
|---|---:|
| Format Valid Rate | 100% |
| Accuracy | 96.77% |
| BANKRUPTCY Recall | 0% |
| F2 | 0 |
| False Negative | 33 |

SFT 的混淆矩阵：

| 实际 / 预测 | BANKRUPTCY | HEALTHY |
|---|---:|---:|
| BANKRUPTCY | 0 | 33 |
| HEALTHY | 0 | 990 |

### 前端建议截图

可以在前端搜索并查看：

```text
finance-risk-sft.F2yWZt5L
```

建议截图：

- 实验详情页；
- 训练状态和配置；
- 评测结果或相关诊断区域。

### 讲解方式

> SFT 的输出格式是正确的，说明不是格式错误，也不是解析器无法读取。但是模型把 1023 个样本全部判断成了 HEALTHY，一个破产样本都没有识别出来。这是典型的多数类塌缩。

需要强调：

```text
Accuracy 很高，不代表模型真的有用。
```

---

## 第 3 页：GRPO 训练问题

### 建议标题

**GRPO 运行了，但大量 Group 没有提供有效学习信号**

### 前端建议截图

查看实验：

```text
finance-risk-grpo.vzN5HQBu
```

建议截图：

- Group 信号统计；
- Constant Group / Effective Group；
- 某一个 Group 的多条 Response；
- 每条 Response 的 Reward 和 Advantage。

### C1 GRPO 结果

| 指标 | C1 GRPO |
|---|---:|
| Constant Group Rate | 66.25% |
| Effective Group Rate | 33.75% |
| Constant Group 高风险规则命中 | 5 个 Step 中命中 4 次 |
| 正类无 Advantage 规则命中 | 5 个 Step 中命中 3 次 |
| Step5 BANKRUPTCY 输出 | 4 / 64 |

### 如何解释 Group

同一个 Prompt 生成 4 个回答，例如：

```text
[0, 0, 0, 0]
```

或者：

```text
[1, 1, 1, 1]
```

这表示 4 个回答的结果完全一样。模型很难从这一组中判断：

- 哪个回答应该被强化；
- 哪个回答应该被抑制。

### 讲解方式

> RL Tool 没有只看平均 Reward，而是把同一个 Prompt 的多个回答重新组成 Group。工具发现，GRPO 中有大量 Group 的 Reward 完全相同，真正包含组内差异的 Group 只有 33.75%。这说明很多训练样本虽然被生成和计算了 Reward，但没有提供足够的相对学习方向。

---

## 第 4 页：根据诊断切换 NGRPO

### 建议标题

**诊断结果指导算法切换，而不是盲目继续训练**

### 前端建议截图

查看实验：

```text
finance-risk-grpo.Cpj70KDp
```

这是 C3 NGRPO 实验。

### 实验控制变量

这一次尽量保持其他条件不变：

- 同一个 SFT checkpoint；
- 同样的训练数据比例 1:3；
- 同样的 Reward；
- 同样的 Group Size；
- 同样的训练 Step；
- 只把 GRPO 换成 NGRPO。

### 对比结果

| 指标 | GRPO C1 | NGRPO C3 | 变化 |
|---|---:|---:|---:|
| Constant Group Rate | 66.25% | 48.75% | 下降 17.5 个百分点 |
| Effective Group Rate | 33.75% | 51.25% | 提升 17.5 个百分点 |
| Constant Group 规则命中 | 4 / 5 | 1 / 5 | 明显减少 |
| 正类无 Advantage 命中 | 3 / 5 | 0 / 5 | 消失 |

### 讲解方式

> 工具发现原来的问题不是简单的 Reward 太低，而是大量 Group 没有组内差异。因此我们切换到 NGRPO。可以看到，有效 Group 比例从 33.75% 提升到 51.25%，说明更多训练样本开始提供可比较的学习信号。

但是要补充：

> 这一步改善的是训练过程，不是最终业务指标。C3 的固定测试集仍然主要预测 HEALTHY，因此只更换算法还不够。

---

## 第 5 页：调整采样并找到 Step1 有效窗口

### 建议标题

**逐 Step 评测，找到真正有效的 checkpoint**

### 实验动作

继续使用 NGRPO，但将训练采样比例从：

```text
BANKRUPTCY : HEALTHY = 1 : 3
```

调整为：

```text
BANKRUPTCY : HEALTHY = 1 : 1
```

同时：

- 每个 Step 保存 checkpoint；
- 每个 Step 进行固定测试集评测；
- 不改变测试集，避免测试标准变化。

### C8 Step1 结果

| 指标 | E1 SFT | C8 Step1 |
|---|---:|---:|
| Recall | 0% | 72.73% |
| F2 | 0 | 18.32% |
| False Negative | 33 | 9 |
| True Positive | 0 | 24 |
| Precision | 0% | 4.59% |
| False Positive | 0 | 499 |

### 讲解方式

> Step1 第一次真正提升了业务指标。33 个破产企业中，模型识别出了 24 个，Recall 从 0 提升到 72.73%，漏报从 33 个降到 9 个，F2 也从 0 提升到 18.32%。

同时说明代价：

> 代价是误报较多，False Positive 达到 499，Precision 只有 4.59%。所以 Step1 不是可以直接上线的模型，而是一个有效的探索窗口。

### 工具价值

如果只等训练结束再评测，可能会错过 Step1。工具做了两件事：

1. 逐 Step 保存 checkpoint；
2. 逐 Step 进行独立测试。

这让研发人员能够及时保留有效窗口，而不是只使用最后一个模型。

---

## 第 6 页：Step2 退化和 checkpoint 拒绝

### 建议标题

**Reward 继续上升，但模型实际退化**

### Step1 和 Step2 对比

| 指标 | Step1 | Step2 | 说明 |
|---|---:|---:|---|
| 训练平均 Reward | 0.3594 | 0.4844 | 上升 |
| Effective Group Rate | 62.5% | 75.0% | 上升 |
| 独立测试 Recall | 72.73% | 100% | 表面上上升 |
| Precision | 4.59% | 3.23% | 下降 |
| F2 | 18.32% | 14.29% | 下降 |
| False Positive | 499 | 990 | 严重上升 |
| KL Loss | 0.000028 | 0.039811 | 快速上升 |
| Grad Norm | 0.189 | 0.580 | 约增至 3 倍 |

Step2 最终把固定测试集的 1023 条样本全部预测为 `BANKRUPTCY`：

```text
Recall = 100%
False Positive = 990
F2 从 18.32% 降到 14.29%
```

### 前端建议截图

在 C8 实验中查看：

- Step1 和 Step2 的训练指标；
- Reward 趋势；
- KL Loss；
- Grad Norm；
- 稳定性告警；
- Rollout 输出比例。

### 讲解方式

> 如果只看训练 Reward，Step2 看起来比 Step1 更好。如果只看 Recall，Step2 也看起来更好。但结合 Precision、F2 和 False Positive 后，我们发现模型从偏向 HEALTHY 变成了偏向 BANKRUPTCY，发生了反方向的单类塌缩。

与此同时：

```text
KL Loss 快速上升
Grad Norm 增长约 3 倍
Entropy 明显变化
Response Clip 持续为 100%
```

这说明策略偏移正在加剧，继续训练风险变大。

最终决策：

```text
保留 C8 Step1
拒绝 C8 Step2
```

---

# 五、最值得放进 PPT 的指标

## 1. 业务效果指标：Recall、F2、False Negative

这是最重要的一组指标。

```text
SFT -> C8 Step1

Recall：0% -> 72.73%
F2：0 -> 18.32%
False Negative：33 -> 9
```

它证明模型从完全漏掉破产样本，变成能够识别大部分破产样本。

但必须同时展示 Precision 和 False Positive，因为 Step1 误报较多：

```text
Precision：4.59%
False Positive：499
```

## 2. 训练信号指标：Effective Group Rate

这组指标体现 RL Tool 如何发现并改善训练过程：

```text
GRPO C1：33.75%
NGRPO C3：51.25%
C8 Step1：62.5%
```

同时：

```text
Constant Group Rate：
66.25% -> 48.75% -> 37.5%
```

可以这样解释：

> 越来越多的 Prompt 能够产生不同质量的回答，训练中真正有用的相对信号增加了。

但需要提醒：

```text
Effective Group Rate 变高，不等于最终业务效果一定变好。
```

Step2 就是这个反例。

## 3. 稳定性指标：KL Loss 和 Grad Norm

这组指标用于判断是否训练过头。

Step1 到 Step2：

```text
KL Loss：0.000028 -> 0.039811
Grad Norm：0.189 -> 0.580
```

简单解释：

- KL Loss：模型和原始策略偏离了多少；
- Grad Norm：每次参数更新的力度有多大；
- 两者同时快速上升，说明策略变化变得激进。

这组指标不是用来证明模型效果提升，而是用来解释：

> 为什么 Step2 虽然 Reward 更高，但不能继续保留。

---

# 六、前端截图建议

## 截图 1：SFT 基线

搜索：

```text
finance-risk-sft.F2yWZt5L
```

用于说明：

```text
格式正确，但模型没有识别出破产样本。
```

## 截图 2：GRPO Group 诊断

搜索：

```text
finance-risk-grpo.vzN5HQBu
```

用于说明：

```text
大量 Group 是全对或全错，Effective Group Rate 偏低。
```

## 截图 3：NGRPO 对比

搜索：

```text
finance-risk-grpo.Cpj70KDp
```

用于说明：

```text
切换 NGRPO 后，Group 学习信号得到改善。
```

## 截图 4：C8 Step1 / Step2

搜索：

```text
finance-risk-grpo.rz7gDzsp
```

用于说明：

```text
Step1 找到有效窗口，Step2 出现策略退化。
```

---

# 七、推荐的 PPT 页面结构

## 第 1 页：案例任务

```text
36 个财务指标
判断 BANKRUPTCY / HEALTHY
重点关注 Recall、F2 和漏报
```

## 第 2 页：SFT 基线问题

```text
Accuracy 96.77%
Recall 0%
F2 0
False Negative 33
```

结论：模型只学会了多数类和输出格式。

## 第 3 页：GRPO 训练诊断

```text
Constant Group Rate 66.25%
Effective Group Rate 33.75%
```

结论：大量 Group 没有有效学习信号。

## 第 4 页：NGRPO 优化

```text
Effective Group Rate：33.75% -> 51.25%
Constant Group Rate：66.25% -> 48.75%
```

结论：训练信号改善，但业务效果还没有立即改善。

## 第 5 页：采样调整和 Step1

```text
Recall：0% -> 72.73%
F2：0 -> 18.32%
False Negative：33 -> 9
```

结论：通过调整采样并逐 Step 评测，找到有效 checkpoint。

## 第 6 页：Step2 退化

```text
Reward：0.3594 -> 0.4844
F2：18.32% -> 14.29%
False Positive：499 -> 990
KL：0.000028 -> 0.039811
```

结论：Reward 上升不等于模型继续变好，应保留 Step1、拒绝 Step2。

## 第 7 页：工具介入闭环

```text
采集日志和 Rollout
        ↓
恢复 Prompt Group
        ↓
发现学习信号不足
        ↓
调整算法和采样
        ↓
逐 Step 独立评测
        ↓
保留有效 checkpoint
        ↓
拒绝退化 checkpoint
```

---

# 八、可以直接照着讲的总结

> 这个案例的任务是让模型根据企业财务指标识别破产风险。
>
> 一开始，SFT 模型虽然输出格式完全正确，Accuracy 也有 96.77%，但实际上一个破产样本都没有识别出来。
>
> 接着我们使用 GRPO 训练。RL Tool 没有只看平均 Reward，而是进一步分析同一个 Prompt 生成的多个回答，发现 66.25% 的 Group 都是全对或全错，真正包含组内差异的 Group 只有 33.75%。这说明很多训练样本没有提供有效的相对学习信号。
>
> 根据这个诊断，我们在保持其他条件不变的情况下切换到 NGRPO。Effective Group Rate 从 33.75% 提升到 51.25%，说明训练信号得到了改善。
>
> 之后我们进一步调整采样比例，并且每个 Step 都保存 checkpoint、进行独立评测。Step1 的 Recall 从 0 提升到 72.73%，漏报从 33 个降到 9 个，F2 也从 0 提升到 18.32%，找到了一个有效的训练窗口。
>
> 但是继续训练到 Step2 后，训练 Reward 继续上升，Effective Group Rate 也继续上升，固定测试集效果却退化了，误报从 499 个增加到 990 个。与此同时，KL Loss 和 Grad Norm 快速上升，说明策略偏移正在加剧。
>
> 所以我们最终保留 Step1，而不是选择 Reward 更高的 Step2。这个案例说明，RL Tool 的作用不是简单展示训练曲线，而是把训练监控、行为诊断、优化实验和 checkpoint 选择连接成一个闭环。

---

## 最终结论

> **RL Tool 让我们不仅知道训练结果是多少，还知道训练过程中发生了什么、下一步应该改什么，以及什么时候应该停止继续训练。**

需要谨慎表达：

- 不要说“NGRPO 单独解决了业务问题”；
- 不要说“训练指标一直变好”；
- 不要说“Step2 比 Step1 更好”；
- 应该说：NGRPO 改善了训练信号，采样调整和逐 Step 评测帮助我们找到 Step1 有效窗口，稳定性监控帮助我们识别并拒绝 Step2 的退化。
