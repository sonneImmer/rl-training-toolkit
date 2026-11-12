> 归档说明（2026-11-12）：本文件保留历史汇报/经验草稿。通用阈值、因果判断和优化建议尚未全部校准，不作为已验证的自动诊断规则；当前状态与限制以`docs/CURRENT_STATUS_20261112.md`及案例指标为准。

# RL 算法选择指南：GRPO vs NGRPO vs DAPO

## 基于 Qwen2.5-7B 金融风控违约预测实验的实证结论

---

## 三种算法核心区别

### GRPO (Group Relative Policy Optimization)
```
advantage_i = (reward_i - group_mean) / group_std
```
- 同一 prompt 的 n 个 rollout reward 做组内归一化
- **依赖组内 reward 方差**：方差大 → 梯度信号强；方差≈0 → advantage≈0 → 无梯度

### NGRPO (Normalized GRPO with virtual max-reward)
```
advantage_i = (reward_i - group_mean_with_virtual_max) / group_std_with_virtual_max
# 向 group 里加入 k 个 max_reward 的虚拟样本后再算均值和方差
```
- 虚拟 max-reward 样本打破同质组退化
- **即使全对/全错的同质组也有有效梯度信号**
- 全对组：加入虚拟+3后均值>真实reward → 真实样本 advantage 为负 → "还不够好"
- 全错组：加入虚拟+3后均值被拉高 → 真实样本 advantage 为负 → "需要改变"

### DAPO (Dynamic Advantage Policy Optimization)
```
GRPO/NGRPO + filter_groups（删除全对组和全错组）
```
- 核心思想：只保留混合组（部分对部分错），因为混合组的对比信号最强
- DAPO 本身不是一个独立的 advantage estimator，而是在 GRPO/NGRPO 基础上叠加数据过滤
- verl 实现中通过 `algorithm.filter_groups` 配置启用

---

## 实验数据

### GRPO vs NGRPO 稳定性对比（相同配置：1:2 数据，±3/±1 reward）

| 指标 | v5 NGRPO | v9 GRPO |
|------|---------|---------|
| Entropy 涨幅 | 0.61→0.88 (1.4x) | 0.62→3.97 (6.4x) |
| KL Loss 终值 | 0.076 | 0.351 |
| pos_recall | 9.62% | 0.96% |

### 全部实验汇总

| 版本 | 算法 | 数据配比 | Reward | pos_recall | neg_accuracy |
|------|------|---------|--------|-----------|-------------|
| v3 | NGRPO | 1:3 | ±3/±1 | 0.96% | 99.61% |
| v4 | NGRPO | 1:1 | ±3/±1 | 52.88% | 49.23% |
| v5 | **NGRPO** | **1:2** | **±3/±1** | **9.62%** | **97.01%** |
| v6 | NGRPO | 1:2 | ±5/±1 | 4.81% | 95.27% |
| v7 | NGRPO | 1:2 | +3/-1 & +1/-1 | 3.85% | 91.62% |
| v8 | GRPO | 1:2 | +3/-1 & +1/-1 | 1.92% | 98.65% |
| v9 | GRPO | 1:2 | ±3/±1 | 0.96% | 98.70% |

---

## 算法选择决策树

```
你的任务 reward 是什么类型？
│
├─ 连续/分级 reward（数学推理、代码生成，partial credit）
│   ├─ group 内 reward 天然多样（std > 0）→ GRPO 够用
│   └─ 同质组较多，GRPO 训练不稳定 → NGRPO
│
├─ 离散/二元 reward（分类、判别、是/否）
│   ├─ group 内同质组占比高 → NGRPO（推荐）
│   └─ 即使 NGRPO 也难以平衡正负 → 调数据配比 + NGRPO
│
└─ 需要聚焦"学习区"数据？
    ├─ 大 n (≥16) + 连续 reward → DAPO (NGRPO + filter_groups)
    └─ 小 n (<16) + 离散 reward → 不建议 DAPO，过滤太激进
```

---

## 各算法适用场景详解

### GRPO 适合的场景
- **数学推理**：同一道题 6-16 个 rollout，有对有错有部分分，group 内 reward 分布丰富
- **代码生成**：通过测试用例数量作为 reward，天然分级
- **关键条件**：group 内 reward 的 std 不能退化到 0

### NGRPO 适合的场景
- **二分类/多分类任务**：reward 只有几个离散值，同质组占比高
- **稀疏 reward**：大部分 rollout 得 0 分，偶尔得 1 分
- **小 n**：n=4-6 时同质组概率极高，NGRPO 虚拟样本是唯一有效的梯度来源
- **任何 GRPO 训练不稳定的情况**：NGRPO 是 GRPO 的安全超集

### DAPO 适合的场景
- **大 n (≥16)**：rollout 数量多，混合组占比高，过滤后有效 batch 不会太小
- **数学/代码推理 + 连续 reward**：有丰富的"学习区"数据
- **与 NGRPO 搭配**：DAPO 过滤同质组 + NGRPO 保证剩余组的梯度质量

### DAPO 不适合的场景
- **小 n (<16) + 离散 reward**：同质组占比极高（二分类 n=6 时，负样本全对概率 ~40%），过滤后有效 batch 大幅缩小
- **正样本本来就稀少**：正样本的"全错组"会被 DAPO 删掉，适得其反
- **数据配比问题**：DAPO 的隐式过滤不如手动调数据配比精准可控

---

## 为什么增大 KL 惩罚不能拯救 GRPO

GRPO 的根本问题不是"跑得太快"而是 **advantage 信号本身有问题**：
- GRPO 在同质组上 advantage≈0，大部分时间传零梯度
- 少量混合组给出的梯度信号不稳定 → entropy 逐渐失控
- 增大 KL 只能用全局正则化限制策略偏离速度，不能解决组间 scale 不一致
- 更强 KL = 学习更慢 = pos_recall 更难提升
- 本质上是用"刹车代替方向盘"，效果不如 NGRPO 的"局部归一化"

---

## 关心指标：如何判断训练是否稳定

| 指标 | 正常范围 | 危险信号 |
|------|---------|---------|
| Entropy 涨幅 | < 2x | > 5x（策略崩溃） |
| KL Loss 终值 | < 0.1 | > 0.3（过度偏离 reference） |
| Response Length 变化 | ±50% | 翻倍或减半（生成模式突变） |
| Clip Ratio | < 0.1 | > 0.3（更新步太大） |

---

## 结论

1. **NGRPO 是分类/判别类任务的首选算法**，虚拟 max-reward 样本解决了同质组梯度退化问题
2. **GRPO 仅适合 group 内 reward 天然多样的任务**（数学、代码），不推荐用于离散 reward 场景
3. **DAPO 是 NGRPO/GRPO 的补充而非替代**，适合大 n + 连续 reward 场景，小 n 离散 reward 下过滤太激进
4. **数据配比的影响远大于算法选择**：1:3→全答否，1:1→全答是，1:2 是当前最佳折中
5. **调参优先级**：数据配比 > 算法选择 (NGRPO>GRPO) > reward 设计 (±3/±1最优) > KL/entropy 超参
