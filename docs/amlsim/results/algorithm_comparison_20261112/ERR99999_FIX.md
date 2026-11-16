# ERR99999 处理记录

## 现象

2026-11-12 的 GRPO、NGRPO、PPO 长训练均完成 `150/150` 并保存了 actor checkpoint，随后在 Ray 收尾阶段出现 `DataLoader worker ... is killed by signal: Killed`，最后输出 `ERR99999 UNKNOWN applicaiton exception`。容器 exit code 为 0，checkpoint 和 rollout 均已生成。

## 定位

错误发生在训练进度完成和 checkpoint 保存之后，不是训练 step 内的 loss、梯度或 NPU 显存错误。历史容器没有 `OOMKilled` 标记，主机也没有对应的内核 OOM 记录。结合 DataLoader worker 收尾堆栈，当前结论是多进程 DataLoader 在 Ray 退出阶段的 worker 清理异常；不能把 ERR99999 解读为模型训练失败。

## 修复

RL 启动链路默认设置 `data.dataloader_num_workers=0`，并把该值写入 `run_manifest.json`。宿主启动脚本将 `DATALOADER_NUM_WORKERS` 透传到训练容器，同时保留环境变量覆盖能力。

## 验证

使用相同镜像、相同数据和 8 卡执行 1-step 冒烟训练：

- 训练完成 `1/1`。
- 容器 exit code 为 `0`，`OOMKilled=false`。
- 未出现 `DataLoader worker ... killed`。
- 未出现 `ERR99999`。
- 生成了 rollout 和训练日志；仅清理了本次冒烟 checkpoint。

该修复已合入 `tools/amlsim/launch_amlsim_rl.sh` 和 `tools/amlsim/run_amlsim_rl.sh`。历史实验日志保持原样，状态仍标记为 `completed_with_errors`，避免覆盖原始证据。
