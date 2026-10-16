# 实验 UI 架构

## 数据流

训练目录是事实源。后台每隔数秒扫描 `train.log` 与 `rollouts/<step>.jsonl`，生成实验快照并写入独立状态目录。浏览器 Dashboard 和 Jupyter Notebook 读取同一份实验对象与诊断结果。

```text
training run -> ExperimentStore -> snapshot/events -> HTTP API -> browser UI
                                             \-> notebook renderer -> Jupyter
```

实验状态与训练产物不写入 Git。Git 只保存解析、规则、服务和前端代码。

## 实验对象

每个实验持久化以下文件：

```text
experiment_registry/experiments/<experiment_id>/
├── manifest.json
├── snapshot.json
├── final_snapshot.json
└── events.jsonl
```

`snapshot.json` 在训练过程中更新；终态实验额外保存 `final_snapshot.json`；标签激活和恢复写入 `events.jsonl`。

## 标签边界

当前规则描述探索收缩、KL 漂移、裁剪饱和、梯度尖峰、输出截断、奖励信号退化和训练性能下降。规则结果明确区分已触发、未触发和未评估，并保留阈值、窗口、证据 step 与建议检查项。规则不推断数据难度、模型能力、算法适配性或因果关系。

## 当前页面

- 实验列表：状态、算法、进度、更新时间和当前标签数。
- 实验详情：实时标签、标准 rollout 质量、有效/低信息 group、informative pair、reward、KL、entropy 和 gradient norm。
- 稳定性规则：展开查看未评估原因、基线、当前窗口、阈值和证据 step。
- 实验对比：查看完整启动参数差异、最新指标差异和对齐 step 趋势。
- Rollout 证据：按 step 查看 prompt、输出、reward、`pred` 和 `acc`。
- Jupyter：在不能开放 Web UI 的环境中展示同一套实验目录、诊断、规则和实验对比。

下一阶段在相同实验对象协议上增加诊断处理记录、父子实验关系和汇报材料导出，不需要修改训练采集链路。
