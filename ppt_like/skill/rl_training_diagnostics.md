> 归档说明（2026-11-12）：本文件为历史经验草稿。通用阈值、因果判断和优化建议未全部校准，不作为已验证的自动诊断规则；当前状态见docs/CURRENT_STATUS_20261112.md。

# RL/GRPO Training Diagnostics Skill

## 适用场景
PPO/GRPO/DAPO/NGRPO 等 RL 微调 LLM 的训练过程监控、问题诊断、最佳 checkpoint 选择。

## 核心监控指标

| 指标 | 含义 | 健康范围 | 危险信号 |
|------|------|----------|----------|
| `actor/entropy` | 输出分布的熵，衡量多样性 | 0.5-2.0（取决于任务） | >3.0 输出混乱；<0.3 可能坍缩到固定模式 |
| `actor/kl_loss` | 与 ref policy 的 KL 散度 | <0.1 初期；<0.5 中期 | >1.0 严重偏离 ref，可能 reward hacking |
| `response_length/mean` | 平均生成长度 | 与初始值偏差 <30% | 快速下降 >50% = 模型学会用短回复投机 |
| `critic/score/mean` | 平均 reward | 逐步上升或稳定 | 持续下降或异常波动加大 |
| `actor/grad_norm` | 梯度范数 | 稳定，不持续增长 | 持续增长 = 训练不稳定 |
| `actor/pg_clipfrac` | PPO clip 比例 | <5% | >10% 更新步太大，需要降 lr 或增 mini-batch |
| `actor/pg_loss` | 策略梯度损失 | 围绕 0 小幅波动 | 持续大幅负值 = 可能过度优化 |

## 典型训练三阶段模式

### Phase 1: 稳定期（通常 epoch 0）
- entropy 低（~0.5-0.8），模型遵循 base policy
- kl_loss 极小（<0.05）
- response_length 稳定，接近初始值
- score_mean 波动大但幅度正常
- **状态**：模型刚开始从 reward signal 学习，行为变化缓慢

### Phase 2: 演变期（通常 epoch 1）
- entropy 逐步上升（1.0-3.0）
- kl_loss 温和增长（0.05-0.3）
- response_length 基本稳定或略有变化
- score_mean 趋势向好或稳定
- **状态**：模型在正常学习，策略逐步偏移 ref

### Phase 3: 崩溃期（通常 epoch 2+）
- entropy 飙升（>5.0）
- kl_loss 暴增（>1.0）
- **response_length 急剧下降**（最关键信号，50%+ 跌幅）
- score_mean 无改善甚至恶化
- grad_norm 可能异常
- **状态**：reward hacking / mode collapse，模型学会用极短/无意义回复投机

## 关键诊断规则

1. **response_length 骤降是最可靠的崩溃信号**
   - 比熵更能反映"模型在投机"
   - 下降 >30% 需警惕，>50% 基本确认崩溃
   - 原因：模型发现短回复可以规避被惩罚的概率

2. **kl_loss 持续 >1.0 意味着策略严重偏移**
   - 超过这个阈值，PPO 的信任域约束已失效
   - 需要增大 kl_coef 或提前停止

3. **entropy 持续上升 + score 不升反降 = reward hacking**
   - 模型在"探索"但没找到有效策略
   - 输出变得随机混乱

4. **score_mean 长期为负 = 训练信号不足或 reward 设计有问题**
   - 对于非对称 reward（正负样本奖惩不等），score_mean 可能自然偏负
   - 需结合正负样本比例分析

## 最佳 Checkpoint 选择原则

1. **选崩溃前的最后一个稳定 step**
2. 具体判据（按优先级排序）：
   - response_length 仍在正常范围（与初始值偏差 <20%）
   - kl_loss < 0.3（偏移可控）
   - entropy 1.0-2.0（有学习但不过度）
   - score_mean 相对最优
3. **通常最佳 checkpoint 在 epoch 1 的某个位置**，而非训练结束
4. 宁可选 score 稍低但稳定的 checkpoint，不要选 score 高但指标异常的

## 常见干预措施

| 问题 | 干预 |
|------|------|
| entropy 爆炸 | 增大 entropy_coeff；减小 lr；减少训练 epoch |
| kl_loss 过大 | 增大 kl_coef；启用 kl_loss（如未启用）；early stop |
| response_length 骤降 | 降低 max_response_length 上限约束；增加长度惩罚 reward |
| score 偏负不改善 | 检查 reward 设计；调整正负样本比例；增大正样本 reward |
| grad_norm 异常 | 降低 lr；增大 ppo_mini_batch_size；gradient clipping |

## 数据采样方式

从 Ray 训练日志提取关键指标：
```bash
# 提取所有 step 的核心指标
grep "^step:" <log_file> | awk -F' - ' '{
  for(i=1;i<=NF;i++) {
    split($i, kv, ":")
    gsub(/^ +/, "", kv[1])
    if(kv[1]=="step") step=kv[2]
    if(kv[1]=="training/epoch") epoch=kv[2]
    if(kv[1]=="actor/entropy") entropy=kv[2]
    if(kv[1]=="actor/kl_loss") kl_loss=kv[2]
    if(kv[1]=="actor/pg_loss") pg_loss=kv[2]
    if(kv[1]=="critic/score/mean") score_mean=kv[2]
    if(kv[1]=="actor/grad_norm") grad_norm=kv[2]
    if(kv[1]=="response_length/mean") resp_len=kv[2]
  }
  printf "step=%3s epoch=%s entropy=%.2f kl_loss=%.3f pg_loss=%.4f score_mean=%.3f grad_norm=%.3f resp_len=%.0f\n", step, epoch, entropy, kl_loss, pg_loss, score_mean, grad_norm, resp_len
}'
```
