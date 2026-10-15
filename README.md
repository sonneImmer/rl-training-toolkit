# RL Training Toolkit

面向 VERL 强化学习训练的可复用补丁与可视化工具。

- `patch_tools/`：在运行中容器内给 VERL 增加 NGRPO，支持应用、检查和恢复。
- `patches/`：让 rollout 输出包含稳定分组信息的最小静态补丁。
- `src/rl_diagnostics/`：解析日志和 rollout，分析组内信号、回答质量与训练稳定性，并输出 HTML/SVG。
- `notebooks/`：适用于只有 Jupyter/终端环境的按 step 实时监控页面。

本仓库不保存模型、数据集、训练日志、rollout 原文或实验产物。

## 安装

```bash
python3 -m pip install -e .
```

开发测试：

```bash
python3 -m pip install -e '.[dev]'
pytest -q
```

## NGRPO 运行时补丁

```bash
python3 patch_tools/patch_verl_ngrpo.py apply \
  --verl-root /workspace/verl \
  --backup-dir /workspace/verl_ngrpo_backup

python3 patch_tools/patch_verl_ngrpo.py check \
  --verl-root /workspace/verl
```

训练时增加：

```text
algorithm.adv_estimator=ngrpo
algorithm.max_reward=1.0
algorithm.add_max_reward_num=1
```

`max_reward` 必须等于奖励函数能产生的实际最高分。恢复原文件：

```bash
python3 patch_tools/patch_verl_ngrpo.py restore \
  --backup-dir /workspace/verl_ngrpo_backup
```

## 离线诊断与图表

```bash
rl-diagnostics analyze \
  --log /path/to/train.log \
  --rollout-dir /path/to/rollouts \
  --output-dir /path/to/diagnostics_output \
  --export-svgs
```

输出包含结构化 CSV/JSON、HTML 报告和可用于汇报的 SVG 图表。输入与输出目录默认由 `.gitignore` 排除。

## Jupyter 实时监控

打开 `notebooks/rl_rollout_live_monitor.ipynb`，设置 `RUN_DIR` 后运行。可观察有效回答率、正奖励率、全对/全错/混合组、reward、截断率、KL、entropy、gradient norm、advantage 和最新回答抽样。
