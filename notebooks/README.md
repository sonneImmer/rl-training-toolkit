# Jupyter rollout monitor

打开 `rl_rollout_live_monitor.ipynb`，设置：

- `RUNS_ROOT`：包含多个训练运行目录的根目录。
- `STATE_DIR`：实验快照保存目录，应放在 Git 仓库外。
- `SELECTED_EXPERIMENT_ID`：需要查看的实验 ID，首次可保持 `None`。
- `COMPARE_EXPERIMENT_ID`：可选的对比实验 ID。

监控器读取 `train.log` 和 `rollouts/<step>.jsonl`，在每个训练 step 完成后刷新；它不逐 token 显示生成过程。页面包含实验目录、rollout 质量、组内学习信号、稳定性规则展开说明、实验参数差异、对齐趋势和 rollout 原始证据。

Notebook 不需要 `ipywidgets`、pandas 或绘图库，只使用 Python 标准库和 Jupyter 自带的 IPython。

如果从源码目录直接运行而未安装包，请从仓库根目录启动 Jupyter：

```bash
jupyter lab
```

使用 NGRPO 前，先在容器中应用运行时补丁：

```bash
python3 patch_tools/patch_verl_ngrpo.py apply \
  --verl-root /workspace/verl \
  --backup-dir /workspace/verl_ngrpo_backup
```

训练参数：

```text
algorithm.adv_estimator=ngrpo
algorithm.max_reward=1.0
algorithm.add_max_reward_num=1
```

`max_reward` 必须与奖励函数实际最高分一致。
