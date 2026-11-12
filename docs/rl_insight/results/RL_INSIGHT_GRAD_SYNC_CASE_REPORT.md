# RL-Insight Actor Update 显存故障注入与优化对照报告

## 1. 报告目的

本报告记录一次受控 RL 性能诊断案例：通过人为注入延迟梯度同步，模拟 Actor backward 阶段显存峰值异常；随后恢复每个 micro-batch 正常同步，验证显存峰值是否下降，并确认 RL-Insight 能否提供定位和复验所需的证据。

本案例的重点不是证明“显存高就一定是梯度同步故障”，而是演示以下诊断闭环：

1. 发现显存异常。
2. 判断异常发生在哪个训练阶段。
3. 根据阶段和配置提出故障假设。
4. 只修改一个变量进行对照实验。
5. 根据优化后结果验证或否定该假设。

## 2. 实验对象

| 项目 | 优化前 | 优化后 |
|---|---|---|
| 实验 ID | `deferred.sSL1j9K6` | `per_micro_batch.5qh7zggG` |
| 梯度同步模式 | `deferred`，人为延迟同步 | `per_micro_batch`，每个 micro-batch 正常同步 |
| 算法 | GRPO | GRPO |
| 模型、数据、卡数 | 相同 | 相同 |
| 随机种子及主要参数 | 相同 | 相同 |
| 训练步数 | 2 steps | 2 steps |
| 实验时间（北京时间） | 2026-11-03 15:16:55 至 15:25:28 | 2026-11-03 15:25:55 至 15:34:33 |

除梯度同步模式外，两组实验保持相同配置。因此，本案例可以用于分析梯度同步方式和显存峰值之间的关系。

## 3. 实验结果

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| VERL Actor 分配峰值 | 61.90 GiB | 50.97 GiB | 下降 10.93 GiB，约 17.7% |
| Actor Update 平均耗时 | 54.09 s | 57.84 s | 未改善 |
| Step 平均耗时 | 122.13 s | 126.64 s | 未改善 |
| 平均吞吐 | 225.78 | 218.19 | 未改善 |

本轮优化主要降低了 Actor backward 的显存风险，没有证明训练速度得到提升。

## 4. Dashboard 组织方式

两组实验使用的是同一个 Grafana dashboard 模板：

- Dashboard：`verl_trainer_v1_with_vllm_engine`
- Dashboard UID：`a4cabbce-92af-4a84-9399-4deba24ae6d1`

它们不是两个独立 dashboard，而是同一个 dashboard 在不同实验标签和不同时间窗口下展示的数据。

数据区分方式如下：

- Trainer 指标和 Tempo 轨迹：通过 `experiment_name` 区分。
- NPU Exporter 和 vLLM Prometheus 指标：主要通过实验时间窗口区分。
- 两组实验按顺序运行，没有同时占用 NPU，因此可使用互不重叠的时间窗口隔离曲线。

### 优化前 Dashboard

```text
http://172.60.6.4:3000/d/a4cabbce-92af-4a84-9399-4deba24ae6d1/verl-trainer-v1-with-vllm-engine?orgId=1&from=1793690205000&to=1793690740000&var-project=rl_insight_memory_case&var-experiment_name=deferred.sSL1j9K6&var-vllm_model_name=model&var-workerid=0&var-replica=All
```

### 优化后 Dashboard

```text
http://172.60.6.4:3000/d/a4cabbce-92af-4a84-9399-4deba24ae6d1/verl-trainer-v1-with-vllm-engine?orgId=1&from=1793690745000&to=1793691285000&var-project=rl_insight_memory_case&var-experiment_name=per_micro_batch.5qh7zggG&var-vllm_model_name=model&var-workerid=0&var-replica=All
```

## 5. 必看图表

以下图表是本案例形成诊断结论的核心证据，建议按顺序查看和截图。

### 图表 1：state timeline

**位置标识：** 面板标题 `state timeline`，Panel ID `40`。

**重点查看：**

- `trainer_step`：完整训练 step 的时间范围。
- `trainer_0`：`gen`、`old_log_prob`、`ref`、`adv`、`update_actor` 等阶段。
- `vllm_replica_0`、`vllm_replica_1`：两个推理副本的 `vllm_generate`、`wake_up`、`sleep`。
- `npu_rank_0` 至 `npu_rank_7`：各 NPU 上的 `actor_update`、`actor_compute_log_prob`、`ref_compute_log_prob`。

**用于证明：** 显存峰值对应的时间段属于 `actor_update/backward`，而不是 rollout 推理、模型加载或 checkpoint 保存阶段。

**截图建议：** 优化前和优化后各截一张，保持相近缩放比例，确保 `vllm_replica_0/1`、`trainer_0` 和 `npu_rank_0..7` 同时可见。

### 图表 2：Trainer Metric - `perf/max_memory_allocated_gb`

**位置标识：** 面板标题为动态的 `$trainer_metric`；在页面变量中选择 `perf/max_memory_allocated_gb`。

**重点查看：**

- 优化前 Step 2：约 `61.90 GiB`。
- 优化后 Step 2：约 `50.97 GiB`。

**用于证明：** VERL 在 Actor 更新过程中记录的瞬时分配峰值下降约 `17.7%`。

**注意：** 这是确认本案例优化效果的主要数值，不应只使用 NPU Exporter 的 10 秒采样曲线代替。

### 图表 3：NPU HBM Used Memory

**位置标识：** 面板标题 `NPU HBM Used Memory`，Panel ID `331`。

**重点查看：**

- 八张 NPU 的 HBM 曲线是否在 Actor Update 阶段同步抬升。
- 优化前曲线的整体峰值是否高于优化后。
- 是否存在单卡明显偏离其他卡的情况。

**用于证明：** 从硬件侧确认显存压力确实发生，并观察八卡之间是否均衡。

**边界：** NPU Exporter 默认按周期采样，可能错过短时尖峰，因此该图用于观察趋势，不单独用于计算最终降幅。

### 图表 4：NPU HBM Utilization

**位置标识：** 面板标题 `NPU HBM Utilization`，Panel ID `330`。

**重点查看：** 优化前 Actor Update 阶段是否更接近显存容量上限，优化后是否获得更大的安全余量。

**用于证明：** 说明优化降低了 OOM 风险和显存水位，而不代表计算速度一定提高。

### 图表 5：NPU AI Core Utilization

**位置标识：** 面板标题 `NPU AI Core Utilization`，Panel ID `329`。

**重点查看：**

- Actor Update 阶段各卡利用率是否一致。
- 显存下降后 AI Core 利用率是否出现明显退化。

**用于证明：** 排除“通过让计算停顿或大幅降低计算负载换取显存下降”的简单解释。

### 图表 6：vLLM Token Throughput

**位置标识：** 面板标题 `vLLM: Token Throughput`，Panel ID `1`。

**重点查看：** `replica=0` 和 `replica=1` 的推理吞吐是否同时存在，两个副本是否明显失衡。

**用于证明：** rollout 推理链路正常工作，显存异常不是因为某个 replica 未工作或请求全部集中到一个 replica。

### 图表 7：vLLM Scheduler State

**位置标识：** 面板标题 `vLLM: Scheduler State`，Panel ID `7`。

**重点查看：** running、waiting 请求数量，以及两个 replica 是否出现长时间排队。

**用于证明：** 排除严重推理排队导致 step 时间异常的可能性。

### 图表 8：vLLM Queue Time

**位置标识：** 面板标题 `vLLM: Queue Time`，Panel ID `11`。

**重点查看：** 优化前后是否出现显著不同的排队延迟。

**用于证明：** 梯度同步优化针对 Actor backward 显存，而不是通过改变 rollout 排队行为获得结果。

### 图表 9：vLLM Requests Prefill and Decode Time

**位置标识：** 面板标题 `vLLM: Requests Prefill and Decode Time`，Panel ID `12`。

**重点查看：** Prefill 和 Decode 时间是否基本处于相似范围。

**用于证明：** 两组 rollout 推理负载具有可比性。

## 6. 推荐 PPT 截图组合

### 核心截图

1. 优化前 `state timeline`。
2. 优化后 `state timeline`。
3. `perf/max_memory_allocated_gb` 优化前后对比。
4. `NPU HBM Used Memory` 优化前后对比。

### 辅助截图

5. `vLLM: Token Throughput`，显示两个 replica 均工作。
6. `vLLM: Scheduler State` 或 `vLLM: Queue Time`，排除推理拥塞。
7. `NPU AI Core Utilization`，说明优化没有简单关闭计算。

如果 PPT 只能放一页，建议保留：优化前时间线、优化后时间线、Actor 峰值对比和一句边界说明。

## 7. 诊断结论

本案例不能仅依据“显存最高使用率”推断故障来源。故障定位依赖以下证据链：

1. RL-Insight 显示高显存发生在 Actor Update 阶段。
2. 优化前后只修改梯度同步模式。
3. 延迟同步模式下 Actor 峰值为 `61.90 GiB`。
4. 恢复每个 micro-batch 正常同步后峰值降至 `50.97 GiB`。
5. 峰值下降方向与“延迟同步导致梯度保留时间变长”的机制一致。

因此可以得出：在本次受控实验中，梯度同步方式是 Actor backward 显存峰值差异的直接变量，RL-Insight 能够帮助定位异常阶段并验证优化效果。

## 8. 结论边界

本报告可以证明：

- RL-Insight 能发现 Actor Update 阶段的显存压力。
- 受控延迟梯度同步会提高本环境中的 Actor 显存峰值。
- 恢复正常同步后，Actor 峰值下降约 `17.7%`。
- 工具可以形成“发现、定位、修改、复验”的性能诊断闭环。

本报告不能证明：

- 任意显存峰值都由梯度同步导致。
- 看到 `61.90 GiB` 就能自动识别故障来源。
- 优化后训练速度一定更快；本轮耗时和吞吐没有改善。
- 其他模型、batch size、FSDP 版本或 NPU 环境会得到相同绝对数值。
- 原始镜像本身存在该缺陷；优化前故障是人为注入的受控故障。

真实客户环境出现类似问题时，仍需结合 OOM 日志、训练配置、阶段时间线、通信行为和必要的 profiler 数据进一步确认。

## 9. 本地材料位置

- 优化前实验：`/mnt/local/transformer_files/zkx/runs/rl-insight-memory-20261103/deferred.sSL1j9K6`
- 优化后实验：`/mnt/local/transformer_files/zkx/runs/rl-insight-memory-20261103/per_micro_batch.5qh7zggG`
- 对比图表：`/mnt/local/transformer_files/zkx/runs/rl-insight-memory-20261103/complete-report`
- 浏览器报告：`http://172.60.6.4:8767/`
