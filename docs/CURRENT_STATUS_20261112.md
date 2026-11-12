# 开发进展、功能与实验总览

核对日期：2026-11-12。用途：为功能分析提供当前实现、证据、缺陷及下一步验收依据。

本轮只整理、验证和归档现有代码与实验，没有重新训练，也没有把本页发现的问题写成已修复。

## 1. 当前结论

- 已形成日志/Rollout采集、Group恢复、学习信号分析、规则提醒、UI/Jupyter展示、报告与实验导出的工具链。
- 已完成企业破产风控原型、AMLSim语义合成反洗钱短训练及150步GRPO/NGRPO对比；FinQA定位为数值推理链路基准。
- 已实现七类稳定性规则；规则存在不等于已完成全部异常的真实故障验证，也不等于自动根因定位。
- RL-Insight已通过运行时补丁接入训练阶段、Worker/NPU及vLLM轨迹；性能案例验证了显存峰值下降，未验证训练加速。
- 当前主要缺口是规则准确性、问题到优化措施的匹配、效果/性能证据统一及更强的实验泛化验证。

## 2. 功能清单

| 功能 | 当前实现 | 入口及边界 |
|---|---|---|
| 动态参数与日志解析 | 提取算法、模型、数据、训练参数和Step指标 | `src/rl_diagnostics/parsers/log_parser.py`；缺失配置不猜测 |
| Rollout与Group恢复 | 按Step/uid关联Prompt与多条Response | `parsers/rollout_parser.py`；原始问题和稳定uid需训练侧输出 |
| 回答质量 | 检查缺失/异常奖励、空输出、截断等事实 | `analyzers/rollout_quality.py`；不进行通用语义推理评分 |
| 组内学习信号 | Reward方差、信息样本对、全对/全错与近极端Group | `analyzers/group_signal.py`；全对/全错需要明确正确性字段，不能通用地从Reward符号推断 |
| Advantage与更新证据 | 补丁记录advantage、更新选择状态及mask/policy-signal token数 | `patch_tools/patch_verl_rollout_evidence.py`；历史数据缺字段无法补成真实记录 |
| 稳定性规则 | 七类规则、正式窗口及至少3个Step的早期观察 | `analyzers/stability_analyzer.py`、`experiment_store.py`；早期观察为低置信度 |
| 异常Prompt与证据 | 从异常Group查看原题、回答、奖励和相关Step | `exporters/prompt_exporter.py`及Web证据区；不等同于自动确认Prompt有错 |
| 分层分析 | 使用已有difficulty/业务标签进行统计 | `analyzers/difficulty_analyzer.py`；没有标签时跳过，不生成通用难度解释 |
| 独立评测 | 接入已完成评测结果，支持类别退化对照 | `tools/attach_evaluation.py`、`evaluation_store.py`；不自动训练/合并/运行checkpoint |
| 实验对象与比较 | 实时刷新、历史快照、参数比较、趋势、原始证据 | `dashboard.py`、`experiment_store.py`、`web/` |
| 阅读交互 | 详情阅读时暂停自动重建；趋势终点选择和悬浮数值 | `web/app.js`；本轮仅语法检查，浏览器交互仍需人工验收 |
| Jupyter | 复用诊断数据展示实验、规则、趋势和Rollout | `notebooks/`、`notebook_view.py`；不承诺Web新增交互全部自动迁移 |
| 报告与迁移 | HTML/SVG/CSV/JSON输出，实验包导入导出并校验哈希 | `reporters/`、`tools/experiment_bundle.py`；Git未保存全部原始实验包 |
| NGRPO注入 | Python补丁/完整静态patch、检查与恢复 | `patch_tools/patch_verl_ngrpo.py`、`patches/002-ngrpo-algorithm.patch` |
| RL-Insight | Trainer/Worker/vLLM埋点，Grafana/Prometheus/Tempo链路 | `patch_verl_rl_insight*.py`及[性能案例](rl_insight/README.md)；额外依赖独立部署 |
| 优化建议 | 已有固定排查提示、人工实验与经验文档 | 尚未形成自动匹配措施、自动复训和自动验收的闭环 |

以上路径未注明前缀时，以`src/rl_diagnostics/`为起点。

## 3. 七类异常的覆盖与缺陷

| 异常 | 使用指标 | 已实现判断 | 当前限制 |
|---|---|---|---|
| 探索收缩 | actor/entropy | 近期与基线窗口比较，默认变化阈值50% | 正式实现误用了绝对变化量，熵上升可能误报；entropy_coeff也被错误列为候选 |
| KL漂移 | actor/ppo_kl、actor/kl_loss | 绝对值中位数连续超过基线3倍 | 两指标应分别解释，不能统一称为相对参考模型偏移；零基线时规则会跳过 |
| Clip饱和 | actor/pg_clipfrac | 连续窗口超过20% | 阈值仍需不同任务校准，不能单独确定根因 |
| 梯度尖峰 | actor/grad_norm | 历史中位数+5倍MAD | 基线接近零时需复核告警意义；不能反推单Group梯度 |
| 输出截断 | response_length/clip_ratio | 超过20%或比基线上升10个百分点 | 达到长度上限不直接等于回答错误 |
| 奖励区分度退化 | low_information_group_ratio等 | 低信息Group超过80%或信息样本对比例下降 | 组内同Reward在NGRPO下不必然表示无Advantage，应结合实录信号 |
| 性能退化 | perf/throughput、timing_s/* | 同阶段基线对比，默认20% | 当前共用了下降判断；耗时应按上升判断，部分键名与实际日志还需适配 |

规则配置为实验版`0.1.0-exp`，并非已校准的业务标准。UI部分性能解释误用了独立评测/业务退化话术，需要与运行性能分开。

## 4. 实验及证据

### 4.1 数学链路与补丁冒烟

已提供`tools/verified_math/`预检、题目恢复、短训练与汇总脚本，并有NGRPO补丁测试。用于验证训练输入、Rollout及patch链路；不作为长训练效果或跨任务泛化证明。恢复入口见[环境恢复](../environment/RECOVERY.md)。

### 4.2 Taiwanese Bankruptcy企业风控

对象：企业36维财务特征、破产/健康分类。已归档SFT、GRPO、NGRPO、采样调整及C8逐Step评测。

- SFT格式有效率100%，但破产Recall为0：格式合规不等于业务有效。
- GRPO与NGRPO短训练对比中，同质Group高风险规则命中从4/5步到1/5步，正类无组内信号规则从3/5步到0/5步。
- 调整采样后的C8 step1破产Recall为72.73%、漏报从33到9，但Precision只有4.59%；step2再次出现单类退化。
- 这些实验包含不同优化变量，不能把采样改善全部归因于NGRPO，也未达到业务验收。

入口：[案例报告](finance_risk/CASE_REPORT.md)、[规则口径](finance_risk/ROLLOUT_RULE_EXPLANATION.md)、[复现](finance_risk/REPRODUCTION.md)。

### 4.3 FinQA

完成数据转换、可执行程序Reward、SFT/RL脚本及冒烟清单。属于金融数值推理，不是风控分类；当前没有最终业务优化结论。入口：[FinQA](finqa/README.md)。

### 4.4 AMLSim语义合成反洗钱

数据由本项目按AMLSim拓扑语义和显式规则生成，不是银行真实交易，也不是直接使用AMLSim原生模拟输出。四类为CYCLE、NORMAL、FAN_IN、FAN_OUT；固定独立测试集800条，各类200条。专家经验为人工定义的生成与判定规则，没有在线专家模型评分。

| 实验 | 风险准确率 | 类型准确率 | 证据F1 | 误报 | 漏报 |
|---|---:|---:|---:|---:|---:|
| SFT step30 | 89.750% | 88.750% | 88.194% | 82 | 0 |
| GRPO step5 | 96.875% | 96.625% | 96.350% | 25 | 0 |
| NGRPO step5 | 98.625% | 93.875% | 98.255% | 11 | 0 |
| GRPO step150 | 99.250% | 99.250% | 99.222% | 0 | 6 |
| NGRPO step150 | 100.000% | 99.875% | 99.961% | 0 | 0 |

短NGRPO的FAN_OUT类型准确率只有82.5%，低于GRPO的99.0%，说明总体风险指标会掩盖业务类型退化。150步结果说明在当前生成分布上继续训练有收益，不能证明真实金融业务泛化或NGRPO普遍占优。当前仍缺多seed和分布外规则测试。

入口：[数据与专家规则](amlsim/EXPERT_RULES_AND_DATASET.md)、[实验协议](amlsim/EXPERIMENT_PROTOCOL.md)、[长短训练对比](amlsim/LONG_TRAINING_COMPARISON_20261031.md)、[复现](amlsim/REPRODUCTION.md)。本轮补充12份长训练评测JSON及两份运行清单。

### 4.5 RL-Insight受控性能案例

两组实验各2步，同环境对比人为延迟梯度同步与每micro-batch同步。Actor分配峰值61.90→50.97 GiB（下降约17.7%）；Step耗时122.13→126.64秒，吞吐225.78→218.19，未提速。短实验只支持本环境的受控机制验证。

同一Grafana模板按实验标签和时间段区分两组；Trainer/vLLM/NPU轨迹用于阶段关联，NPU Exporter周期采样可能漏掉瞬时显存峰值。仅凭显存曲线不能确认故障根因，也未证明完整HCCL算子级通信分析。

入口：[集成与复现边界](rl_insight/README.md)、[完整报告与必看图表](rl_insight/results/RL_INSIGHT_GRAD_SYNC_CASE_REPORT.md)、[摘要与图表](rl_insight/results/REPORT.md)。

## 5. 功能分析建议与下一步验收

| 优先级 | 工作 | 验收依据 |
|---|---|---|
| P0 | 修正规则方向、指标语义与别名 | 熵上涨不触发探索收缩；entropy_coeff不当作熵；耗时上涨/吞吐下降都能正确识别；PPO KL与参考KL分别说明 |
| P0 | 校准低信息Group与真实Advantage的关系 | 同Reward但非零Advantage的NGRPO样例不误写成无学习；缺记录明确显示未知 |
| P1 | 从证据匹配优化建议 | 针对已支持问题输出具体数据/Reward/算法/参数措施、适用条件与复验指标；避免通用阈值直接归因 |
| P1 | RL-Insight复现与统一关联 | 外部依赖版本、启动和dashboard配置可恢复；按experiment/step/rank/phase关联效果与性能证据 |
| P1 | 优化前后效果与性能对照 | 固定主要变量，独立评测与耗时/显存/吞吐分别验收，未改善如实记录 |
| P1 | 提高实验可信度 | 多seed、难度与规则分布变化、真实或独立来源风控数据，记录置信区间及业务分类指标 |
| P2 | 展示与文档一致性 | Web/Jupyter逐项核对；人工验收Rollout阅读不刷新、趋势筛选及比较交互 |
| P2 | 历史经验审查 | PPT和skill文档标注历史经验/待验证建议，清除无证据的通用阈值与因果断言 |

当前已形成可分析、可复盘的原型；“自动根因定位、自动优化、生产风控验收、跨环境一键恢复全部数据”均未完成。

## 6. 本轮验证与归档范围

- 2026-11-12执行`PYTHONPATH=.test-deps:src python3 -m pytest -q`：335通过、1失败。失败项为`test_generator_records_upstream_and_experiment_rule_boundaries`，子进程缺少pandas；未安装新依赖或修改生成器来规避。
- 本轮另检查JSON解析、Python语法、shell语法、Web JavaScript语法和Git差异格式；具体结果见[验证记录](SYNC_VALIDATION_20261112.md)。
- 归档代码、文档、小型指标、运行清单与图表；完整日志、Rollout、权重、镜像、Grafana/Prometheus/Tempo历史库需外部保存。
- 仅克隆GitHub可以恢复工具和结果材料，不能自动恢复原始实验现场。`environment/RECOVERY.md`与`tools/experiment_bundle.py`说明外部资产恢复方法。
- 原始镜像为`openpangu-72b-2512-postrain:v1`；部分历史启动器使用本地派生tag，不能把两者直接当成同一镜像。迁移需核对镜像哈希与全部补丁，当前不新增统一配置系统。
