# 归档索引

本仓库归档的是可复用代码、实验协议、标准化结果和复现说明；模型权重、完整日志、完整 Rollout 和 Parquet 数据仍保存在实验环境之外。

## 已归档案例

- `docs/amlsim/`：基于 AMLSim 拓扑语义的反洗钱规则实验，包含完整结果和 PPT 材料。
- `docs/finance_risk/`：UCI Taiwanese Bankruptcy 风控原型，包含 E0/E1、GRPO/NGRPO、C8 checkpoint 结果和诊断证据。
- `docs/finqa/`：FinQA 可执行金融数值推理工具链，定位为 Reward 和 RL 链路基准，不作为信用风险案例。

## 已归档工具

- `src/rl_diagnostics/`：日志、Rollout、Group、稳定性和实验对象诊断。
- `patch_tools/`、`patches/`：NGRPO 和 Rollout 运行时补丁。
- `notebooks/`：Jupyter 临时监控。
- `tools/amlsim/`、`tools/finance_risk/`、`tools/finqa/`：数据准备、Reward、预检、训练、评测和报告脚本。

## 状态约定

- “原型完成”：链路已跑通并有结果，但不代表业务指标达标。
- “工具验收”：代码和固定样例通过测试。
- “业务验收”：需要固定测试集、业务门槛和多 seed 复现；当前金融风控案例尚未达到该状态。

## 2026-11-12同步补充

- `CURRENT_STATUS_20261112.md`：功能清单、实验台账、七类规则缺陷与优先级。
- `SYNC_VALIDATION_20261112.md`：本轮验证结果与限制。
- `amlsim/results/evaluations/*-long-step*/metrics.json`：GRPO/NGRPO各6个长训练checkpoint评测快照。
- `rl_insight/`：受控显存案例的脚本说明、最终对照数据、运行清单与标注图表。
- `PPT_CASE_MATERIAL.md`及`ppt_like/`：历史汇报草稿，引用时以本轮总览和原始指标为准。
- `ARCHIVE_MANIFEST_20261112.json`：本轮归档文件清单、大小和SHA256。
