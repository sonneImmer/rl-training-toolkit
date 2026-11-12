# RL-Insight集成与性能案例

核对日期：2026-11-12。

## 已归档实现

- `patch_tools/patch_verl_rl_insight.py`：VERL训练阶段/Actor/Ref等埋点。
- `patch_tools/patch_verl_rl_insight_rollout.py`：vLLM rollout阶段埋点。
- `patch_tools/patch_verl_gradient_sync_case.py`：受控梯度同步故障注入与恢复；不是普通训练必装补丁。
- `tools/rl_insight_memory_case/`：双模式启动、训练、Prometheus查询与报告生成。
- `results/`：2026-11-03最终有效对照报告、CSV、SVG、运行清单与起止时间。试验性失败轮次没有混入最终对照。

## 验证结果

最终实验为`deferred.sSL1j9K6`和`per_micro_batch.5qh7zggG`，各2步。Actor显存峰值61.90→50.97 GiB，下降17.7%；Step耗时和吞吐没有改善。完整阶段边界、截图面板和因果解释见[详细报告](results/RL_INSIGHT_GRAD_SYNC_CASE_REPORT.md)。

Grafana使用同一个dashboard模板，按experiment_name和时间窗口筛选；报告中的IP与历史绝对路径是当时环境地址，迁移时需要替换。图表已单独归档，可以不启动监控后端直接查看SVG/HTML。

## 复现前提

1. 获取外部OpenPangu镜像、Qwen/SFT权重、AMLSim语义合成数据及Ascend运行环境，按`environment/RECOVERY.md`和`docs/amlsim/REPRODUCTION.md`核对。
2. 准备RL-Insight源码、OpenTelemetry依赖、Grafana、Prometheus、Tempo、NPU exporter。本仓库不包含这些第三方安装包或历史时序/trace数据库。
3. 本机使用的RL-Insight源码位于`/mnt/local/rl-insight-official-20261031`，Git基点`c7eb5e902450bba220676042706639240049fee0`。仅记录Git基点不能替代外部工作区改动归档；完整监控栈恢复仍待验收。
4. 配置`MODEL_HOST`、`DATA_HOST`、`RUNS_HOST`、`PATCH_HOST`、`TOOLS_HOST`、`AMLSIM_TOOLS_HOST`、`RL_INSIGHT_HOST`、`RL_INSIGHT_OTEL_HOST`、`RL_INSIGHT_SERVER_URL`、`IMAGE`和`HCCL_SOCKET_IFNAME`。启动器默认包含本机路径与派生镜像tag，不能直接用于任意机器。

完成上述准备后，在宿主机分别运行：

```bash
GRAD_SYNC_MODE=deferred TOTAL_STEPS=2 bash tools/rl_insight_memory_case/launch_memory_case.sh
# 等待第一组完成后再运行下一组，避免共享8卡与时间窗口干扰。
GRAD_SYNC_MODE=per_micro_batch TOTAL_STEPS=2 bash tools/rl_insight_memory_case/launch_memory_case.sh
python3 tools/rl_insight_memory_case/build_memory_case_report.py --help
```

每次启动返回容器ID；应检查退出码和`case_timing.json`后再运行下一组。报告脚本通过Prometheus历史查询生成曲线；离线只有本次Git归档图表时，无法重建完整时序数据。

## 待完善

- 监控栈第三方版本、工作区差异、dashboard及无凭据配置的可移植归档。
- 训练实验ID与NPU/vLLM指标的统一标签，减少仅依赖时间窗口隔离。
- 与诊断平台统一关联，及显存、计算、通信瓶颈的更多单变量验证。
- 长时间稳定性、性能重复性和监控开销评估。
