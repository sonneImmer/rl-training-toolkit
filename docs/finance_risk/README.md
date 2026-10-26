# 金融风控 RL 原型案例

本目录归档 2026-10-20 基于 UCI Taiwanese Bankruptcy Prediction 的金融风控原型。任务使用 36 个财务指标预测 `BANKRUPTCY` 或 `HEALTHY`，重点验证 UI 能否从 Rollout group、稳定性指标和独立测试中发现类别塌缩与短暂有效 checkpoint。

## 当前结论

- E1 SFT 格式有效率为 100%，但将全部样本预测为 `HEALTHY`，破产类 Recall 为 0。
- C1 GRPO 的平均 Constant Group Rate 为 66.25%，组内 Reward 差异不足。
- 同条件 NGRPO 运行的 Effective Group Rate 从 33.75% 增至 51.25%，但固定测试集仍无业务提升，不能只凭该结果断言 NGRPO 普遍优于 GRPO。
- 调整为 1:1 训练采样后，C8 step 1 Recall 达到 72.73%、F2 达到 18.32%，但 Precision 仅 4.59%。
- C8 step 2 虽然训练 Reward 上升，却退化为全部预测 `BANKRUPTCY`。该案例证明训练 Reward、Rollout 诊断和独立评测必须联合使用。

## 导航

- `CASE_REPORT.md`：统一实验说明、指标解释和结论边界。
- `PPT_MATERIAL.md`：汇报页素材。
- `REPRODUCTION.md`：数据、SFT、RL、诊断和评测命令。
- `results/`：数据清单、独立指标、运行清单、Rollout 汇总与稳定性 CSV。

## 状态

这是工具验证和检查点选择原型，不是已通过业务验收的风控模型。主要限制是 Precision 低、单 seed、训练步数短、没有专家规则或真实客户数据验证。
