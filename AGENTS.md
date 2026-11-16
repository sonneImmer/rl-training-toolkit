# AGENTS.md — RL Training Toolkit 持续协作说明

此文件供后续在本仓库工作的 Coding Agent / Research Agent 自动发现。**始终优先相信最新代码、实际训练证据与现有测试，而不是旧规划文字。** 说明语言使用中文；技术名词与 API 名称保留原文。

## 任务与长期方向

项目是为 verl 强化学习训练提供 NGRPO 等可回滚补丁、Rollout/Reward/Advantage/稳定性诊断、Web/Jupyter 展示、RL-Insight 关联及可复现实验的独立工具仓库；不是完整 verl Fork。

当前重点是建立 **公开标准数据集（金融、数学、代码）上的强化学习对照**，并把“正确性 → Reward → Advantage → 实际策略更新 → 独立评测 → 计算成本”形成可信证据链。

**详细实施协议与任务状态：** [docs/BENCHMARK_RL_AGENT_EXECUTION_PLAN.md](docs/BENCHMARK_RL_AGENT_EXECUTION_PLAN.md)。它包含数据集、Verifier/Reward 接口、混合组率公式、算法矩阵、8 卡实验要求、里程碑 M0–M6、验收测试和持续交接日志。历史机制方案：[docs/ALGORITHM_MECHANISM_DIAGNOSTICS_ROADMAP.md](docs/ALGORITHM_MECHANISM_DIAGNOSTICS_ROADMAP.md)。

## 每次启动的固定工作步骤

1. 读取本文件、详细计划、`README.md` 和**最新** Git 提交；必要时读取 `docs/CURRENT_STATUS_20261112.md`，但将其视为历史快照。
2. 检查工作树、现有模块、相关测试及失败记录。特别注意：旧版计划中的前端状态丢失及 Entropy/耗时规则错误，在更新版本中已有修复代码迹象，**不能盲目重复修复**。
3. 从计划中选**一个依赖已满足、能够验证的最小任务**；优先修复真实性/正确性/安全，再扩展数据集或新算法。
4. 修改前明确：所需资源、目标文件、是否涉及数据/模型/硬件、验收方式和回滚方案。不要在未经授权时启动占用 8 张 NPU 的长训练、修改敏感环境或消耗外部费用。
5. 尽可能先做单元测试与最小冒烟；执行后记录真实命令、退出码、指标及来源。没有可用 NPU 就只做静态/离线代码验证，并标注未训练。
6. 将实际工作和结果写入详细计划的进度日志，保留失败和阻塞记录；提交时只包含相关代码/测试/文档，不覆盖其他人的更改。
7. 每轮交付应能被另一 Agent 接续：陈述最新提交、已完成/未完成、关键证据路径、下一项最小任务。**“长期执行”指跨会话可接力，不代表后台自动持续运行。**

## 重要现状与源码位置

- `patch_tools/`、`patches/`：verl NGRPO、Rollout 证据和 RL-Insight 埋点；部署前检查 verl/镜像兼容与回滚。
- `tools/finqa/`：FinQA 官方数据转换、程序执行 Reward、SFT/RL 脚本已存在；尚需正式跨算法公开基准。
- `tools/verified_math/` 与 `src/rl_diagnostics/evaluators/math_evaluator.py`：数学输入/答案解析已有基础；复杂 MATH 等价验证不能直接假定正确。
- `src/rl_diagnostics/evaluators/code_evaluator.py`：当前子进程执行**不构成可信沙箱**；正式运行生成代码前必须完成隔离 Worker 与安全验收。
- `src/rl_diagnostics/analyzers/group_signal.py`：Group 区分和实际 Advantage 数据已有；不要新建相互冲突的第二套通用混合率算法。
- `src/rl_diagnostics/algorithm_profiles.py`、`recommendation_rules.py`：GRPO、NGRPO、PPO/GAE 机制诊断已部分实现。
- `src/rl_diagnostics/experiment_store.py`、`dashboard.py`、`web/app.js`：实验快照和页面交互。
- `docs/amlsim/results/algorithm_comparison_20261112/`：150 Step、单 seed、合成规则分布三算法对照；用于 Sanity Check，不代表真实业务泛化。

## 核心数据语义：不可混淆

- `Verifier` 决定**可验证正确性**（`true` / `false` / `null`）；`Reward Policy` 将验证事实映射到**训练分数**，二者解耦。验证失败/沙箱阻断是未知，不能默默记成错误/零分。
- **Mixed Correctness Group Rate** 在逐 Response 正误标签完整、Group 身份可靠且组内至少两条回答时计算：至少有一个对、一个错的组数 / 完整可评估组数。无可靠标签时标 `not_evaluated`。
- **Reward Informative Group Rate** 检查组内 Reward 差异；**Nonzero Advantage Group Rate** 使用实际训练记录的 Advantage。三者不能合并成“有效 Group”。
- `policy_signal_tokens` 仅表示 Mask 内非零 Advantage 的 Token 数，不是真实反向梯度贡献；GSPO 是序列级重要性比率/裁剪，不是逐 Token Reward。
- 不按算法名字推断实现了某论文全部机制：本仓库 NGRPO 主要是虚拟最大 Reward 优势估计；PPO 使用 Critic/GAE，不能用 GRPO 混合组率直接衡量 PPO 是否有梯度。
- 每次实验都固定训练/验证/保留测试 split、数据及 Prompt 哈希、Verifier/Reward/算法版本、初始化权重、seed、资源与预算；测试集不得调参或泄漏进训练。
- 优化结论拆成**模型质量、有效学习信号、端到端效率、资源占用**四项，单独给证据。

## 新任务优先序

1. **M0**：检查现有修复、数据来源、正确性与失败路径，补回归测试。
2. **M1**：建立可复用的 Dataset Adapter → Task Verifier → Reward Policy → Diagnostics 接口和通用混合率。
3. **M2**：复用 FinQA 形成完整公开金融数值推理对照。
4. **M3**：GSM8K 冒烟、MATH 正式数学推理和 MATH-500 保留评测。
5. **M4**：先安全代码沙箱，后 MBPP/APPS 测试和 RL，LiveCodeBench 留作评测。
6. **M5**：TAT-QA、真实信用/破产风控的多 seed、传统基线与类别指标。
7. **M6**：有真实证据后才逐项验证 DAPO、Dr.GRPO、GSPO 等机制，接 RL-Insight 做计算效率对照。

每个阶段的任务前置条件和验收细节以详细计划为准。**若最新代码已实现某项，则检查证据后跳过；不能为了符合计划重复编码。**

## 工程安全与操作边界

- 不自动下载或公开受限数据，不上传密钥、内网配置、原始金融客户样本、大型 checkpoint/权重/完整 Rollout。
- 不直接在宿主 Python 或持有模型/金融数据挂载的容器执行模型生成的任意代码；沙箱不足时`blocked`。
- 没有真实实验就不填写想象成绩；没有完整 Token Log Prob 和 Actor Loss 证据就不做 GSPO Ratio/Clip 的“已测量”陈述。
- 修改实际算法、Reward、训练脚本或生产/共享环境前，先建立单测、可回滚补丁和对照计划；改变 Reward 上界需同步检查 NGRPO 配置。
- 若只能在本次请求里完成部分任务，就如实交付部分结果和后续入口，不声称后台会自行继续。

## 交接记录格式

```text
Task ID:
当前 Git revision / 数据版本:
已修改（含重要设计取舍）:
实际运行测试及结果:
训练或评测证据（若无：未执行）:
遗留问题 / BLOCKED 原因:
下一项最小可验证任务:
提交/PR:
```

**本文件不覆盖 GitHub Actions、部署凭据、NPU 调度权限、或用户对实际长训练的明确授权。**
