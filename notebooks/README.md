# Jupyter rollout monitor

打开 `rl_rollout_live_monitor.ipynb`，将 `RUN_DIR` 改为训练输出目录，然后依次运行单元格。

监控器读取 `train.log` 和 `rollouts/<step>.jsonl`，在每个训练 step 完成后刷新；它不逐 token 显示生成过程。

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
