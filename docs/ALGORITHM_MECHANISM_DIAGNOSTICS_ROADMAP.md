# RL Training Toolkit：算法机制诊断与交互优化实施方案

> 制定日期：2026-11-12  
> 状态：**待实施方案（非已完成功能）**  
> 适用仓库：`sonneImmer/rl-training-toolkit`，以 `main` 分支 2026-11-12 源码为依据。  
> 目标：将“Rollout/Reward/稳定性监控”升级为“**观测事实 → 解释算法机制 → 给出条件性优化建议 → 设计受控复验**”，同时修复已确认的交互和数据口径问题。

## 1. 背景、现状与原则

### 当前已经实现

- verl 运行时补丁：`patch_tools/patch_verl_ngrpo.py`、`patch_tools/patch_verl_rollout_evidence.py`；静态补丁见 `patches/`。
- Rollout 原始字段及部分更新候选证据：`uid`、`score`、`advantage`、`adv_estimator`、`selected_for_update`、`response_mask_tokens`、`policy_signal_tokens`，以及评分器提供时的 `reward_components`。
- 标准化解析、Group 区分度、回答质量、稳定性规则、实验比较、外部独立评测、Web/Jupyter/HTML 报告。
- RL-Insight Trainer/Worker/vLLM 埋点与受控梯度同步案例；现有案例证明特定实验的显存峰值变化，**未证明训练加速**。

详见 [当前状态](CURRENT_STATUS_20261112.md)、[实验 UI 架构](EXPERIMENT_UI_ARCHITECTURE.md)、[RL-Insight 案例](rl_insight/README.md)。

### 本轮设计原则

1. **只从训练实际提供的证据下结论。** Reward 不等于 Advantage；非零 Advantage 不等于真实梯度；训练 Reward 改善不等于独立任务效果改善。
2. **按照实际机制而非算法名称分析。** 优势估计器、重要性比率粒度、Clip 方案、Loss 归一化、Reward/KL 位置、采样/过滤机制分别配置。
3. **新规则先给出假设与可复验操作，不作自动根因归因。** 推荐算法切换必须标记为“候选实验”，不得写成确定性修复。
4. **优先修复 P0 正确性和刷新问题，再扩展指标。** 复用已有解析器、规则和独立评测，避免新建第二套计算口径。
5. **默认不更改实际训练算法，不自动启动训练，不默认逐 Token 全量落盘。** 精细证据采用有界采样、聚合统计和可关闭的埋点。

## 2. 源码审查发现的优先问题

| 优先级 | 已观察到的问题 | 源码定位 | 目标 |
|---|---|---|---|
| P0 | Entropy 探索收缩判断取了绝对变化；`actor/entropy_coeff` 错列为熵指标 | `analyzers/stability_analyzer.py`、`config/stability_config.py` | 只有实际 Entropy 有方向地下降才可触发“探索收缩” |
| P0 | 性能退化规则将吞吐和 `timing_s/*` 统一视作“下降为坏” | 同上 | 吞吐下降、耗时上升分别判断；缺失指标标注未评估 |
| P0 | “训练现象”与“稳定性判断”各有规则实现，可能同名不同定义 | `experiment_rules.py`、`analyzers/stability_analyzer.py` | 统一事件/证据协议，前端按级别展示，不重复计算 |
| P0 | 轮询后重建整页导致独立评测折叠卡片重新展开 | `web/app.js`：`loadExperiments`、`renderDetail`、`evaluationHtml` | 数据继续刷新，卡片、滚动、筛选与阅读位置不被重置 |
| P1 | `selected_for_update` 目前依据 warmup/Mask 推算，未证实 Actor Update 执行完毕；`policy_signal_tokens` 只统计非零 Advantage | `patch_tools/patch_verl_rollout_evidence.py` | 改成准确语义；真正的更新证据需 Actor Update 内埋点 |
| P1 | Live Snapshot 与标准 Rollout Parser 分开计算；原始证据 API 直接读 JSONL | `live_monitor.py`、`parsers/rollout_parser.py`、`experiment_store.py` | 统一指标定义、Group 归属、来源与缺失语义 |
| P1 | 无 `uid` 时回退到 Prompt 指纹，可能把同文不同组误合并；原始证据接口仅返回前 100 条记录 | `parsers/rollout_parser.py`、`dashboard.py`、`web/app.js` | 不确定 Group 显式标记；Group 级完整分页和总数展示 |
| P2 | KPI、6 张图、诊断卡、近期 Step 表重复呈现部分同一指标；不同量纲共用 Y 轴 | `web/app.js` | 去重、指标分组、分轴或拆图，并提供 Step → Group → Response 的联动 |

> 上述是静态源码审查结论，并非修复已经完成，也不能替代真实浏览器回归测试。当前仓库状态文档已记载部分规则问题尚未修复。

## 3. 新模块：算法机制诊断（Algorithm Mechanism Diagnostics）

建议统一划分为四个诊断维度：**学习信号（Reward/Advantage）**、**更新约束（Ratio/Clipping）**、**训练权重（Length/Loss Normalization）**、**采样成本（Filtering/Token/Wall Time）**。

| 机制/方法 | 典型研究问题 | 现在可做 | 需要新增训练证据 | 候选建议与验证边界 |
|---|---|---|---|---|
| GRPO | 组内 Reward 全同导致相对优势不足 | Group 区分度、已记录 Advantage、全对/全错与零信号比例 | 过滤前后 Group 数、有效更新证据 | 检查奖励判别能力、采样温度/任务混合或组大小；用独立评测验证 |
| 现有 NGRPO 实现 | 虚拟最高 Reward 使全错组可能获得负优势 | 比对 Reward 全同与实际非零 Advantage；关注负优势集中 | 正负裁剪与真实更新统计 | 校验 `max_reward`、负优势/类别退化；**现有补丁主要实现优势校准，不应假定包含论文中所有裁剪机制** |
| DAPO | 探索不足、无效组采样、长度与 Token 归一化 | 组信息量、输出长度、Reward 组件（仅初筛） | 正/负方向 Clip、采样尝试/过滤/重试、真实 Loss 分母 | 测试 Clip-Higher、Dynamic Sampling、Token-Level Loss；不要在 NGRPO 下无条件过滤全错组 |
| Dr.GRPO | 组标准差与按回答长度归一化引入权重偏差 | Reward 方差、长度分层及 Advantage（只能做候选分析） | 实际 Loss Reduction、每回答/每 Token 权重 | 固定长度分母/去标准差归一化的单变量对照；不能凭长度直接宣称梯度偏差 |
| GSPO | Token 级重要性比率的训练稳定性 | 识别实际 Loss 机制与历史日志 | 有效 Token 的 old/current Log Prob 或在 Loss 内直接聚合 Sequence Ratio、序列级 Clip | 使用序列级重要性比率和序列级裁剪诊断；**GSPO 不是逐 Token Reward 方法** |
| GAPO（待指定变体） | 同名研究中的自适应裁剪/稳健优势等问题 | 暂不自动建议 | 必须先确定论文、公式、版本与实际启用机制 | 不把不同 GAPO 同名方法合并成单一诊断公式 |

### 关键概念与命名要求

- **Token Reward**：只有评分器或过程验证器确实提供逐 Token/Step 回报，才能称为这一名称。
- **Token Advantage / 策略损失权重**：可以来自序列级 Outcome Reward 的广播及后续 Loss 计算，不能称每个 Token 的独立奖励或真实正确性归因。
- **`policy_signal_tokens`**：现阶段仅表示 Mask 内 `|Advantage| > epsilon` 的 Token 数；不扣除裁剪饱和、零梯度网络效应，不能命名为“有效梯度 Token”。
- **`selected_for_update`**：当前实现是推算型标记，应重命名显示为“更新候选/条件满足”；若要声称“实际更新”，必须从执行路径记录成功的 Actor Update。
- **同 Reward Group ≠ 所有算法中零 Advantage**：NGRPO 可破除部分全错组零优势；KL 等辅助项也可能提供梯度。
- **GSPO Sequence Ratio**：需要训练时对应策略的有效 Token Log Prob；不得仅用 Reward/文本离线推断。
- **GAPO**：必须保存具体论文/机制 ID，例如 `algorithm_profile_id`，不能仅靠字符串“gapo”决定规则。

## 4. 建议新增的数据契约（草案）

先在 Actor Update 计算 Loss 的位置采集**逐步聚合统计**；仅对异常 Group 按配置采样少量可追溯明细。避免默认将所有 Token 的 Log Probability 数组写入 JSONL。

```json
{
  "schema_version": "policy_update_evidence/v1",
  "experiment_id": "...",
  "global_step": 1,
  "actor_update_id": "...",
  "ppo_epoch": 0,
  "mini_batch_id": 0,
  "algorithm_profile": {
    "adv_estimator": "grpo",
    "ratio_scope": "token",
    "clip_mode": "symmetric",
    "loss_reduction": "implementation_specific",
    "kl_location": "loss"
  },
  "source": "actor_loss",
  "sample_count": 16,
  "response_mask_tokens": 1024,
  "nonzero_advantage_tokens": 740,
  "positive_advantage_tokens": 410,
  "negative_advantage_tokens": 330,
  "clip_saturated_positive_tokens": 0,
  "clip_saturated_negative_tokens": 0,
  "ratio_summary": null,
  "sequence_ratio_summary": null,
  "coverage": {
    "ratio": "not_recorded",
    "clipping": "not_recorded",
    "actor_update_executed": "observed"
  }
}
```

此处数量只是**数据格式示例**，不是某次实验的实际值。没有真实埋点时应为 `null` / `not_recorded`，不能写入示例数字冒充观测。

采集与计算要求：

1. 每个采集项记录来源阶段、算法配置和计算公式版本；`global_step` 之外记录 `actor_update_id`、`ppo_epoch`、`mini_batch_id`。
2. 区分 Prompt ID、生成 Group UID、Rollout UID 与尝试批次 ID；缺失或不唯一时明确 `unresolved`。
3. 正负 Clip 饱和按**实际 Loss 分支**定义，而非仅比较 Ratio 是否越界；GSPO 使用序列级裁剪语义。
4. 保存实际 Reduction 分母及有效 Mask Token 总数，支持解释长度偏差；不从空格切词估算精确 Token 数。
5. 过滤前后、重试、有效 Group、实际生成 Token、Wall Time 分开记录；动态采样推荐只在这一数据完整时启用。
6. 保存数据覆盖率、采样比例和是否有真实 Actor 更新记录；缺字段显示 `not_evaluated` 而不是“正常”。
7. 原始 Prompt 和 Response 可能包含敏感金融数据；默认只存聚合与脱敏样本，检查传输和 UI 访问控制。

## 5. 诊断卡与推荐规则（第一期）

诊断卡统一格式：

- **观察事实**：数值、单位、基线、窗口、Step/Group/Response、样本量及来源。
- **机制解释**：仅解释当前实际启用的算法部件与其可能的关系。
- **状态与置信度**：`observed` / `estimated` / `not_recorded` / `not_applicable` / `requires_experiment`；早期窗口低置信度。
- **候选措施**：适用前提、潜在副作用、建议一次只调整的变量。
- **复验方案**：相同数据和 seed/预算，比较训练信号、独立测试质量、时间、显存、吞吐等。

### 第一批建议规则

| 规则 ID | 观察条件（非默认生产阈值） | 输出与限制 |
|---|---|---|
| `group_advantage_collapse` | 可分析 Group 中同质占比高，且记录的零 Advantage 占比高 | 检查 Reward、Group 生成多样性和 GRPO 的相对优势；NGRPO 必须以实际 Advantage 重新判断 |
| `ngrpo_negative_advantage_concentration` | 全错组存在明显负优势，可能伴随类别效果退化 | 核对虚拟最大奖励设定及固定独立评测；不能因非零 Advantage 就宣称有效 |
| `positive_clip_saturation` | 有实际 Actor Loss 证据，正优势上界裁剪集中 | 先核对 Clip 参数、学习率、Ratio 分布，再提出 DAPO 式 Clip-Higher 的单变量对照 |
| `length_weight_imbalance` | 长度分层在已知 Loss 归一化下权重偏斜 | 考虑 Dr.GRPO/DAPO 类归一化对照；必须说明实际公式 |
| `token_sequence_ratio_divergence` | Token Ratio 长尾明显，且序列比率/剪裁统计完整 | 可建议 GSPO 类序列级对照；不直接建议“逐 Token Reward” |
| `low_signal_compute_waste` | Group 信息量较低且消耗可量化生成 Token/时间 | 检查任务组合、采样/动态采样；NGRPO 下区分全错组是否仍有负优势 |

**不设未经实验校准的统一百分比红线。** 规则版本、适用算法、源字段与计算方式必须随诊断一起保存。

## 6. Web/Jupyter：刷新不打断阅读的修复方案（P0）

### 根因（已在现有代码中定位）

- `web/app.js` 每 10 秒执行 `loadExperiments(false)`，部分路径调用 `selectExperiment()`，进而使用 `el("detail").innerHTML = ...` **重建整个详情 DOM**。
- `evaluationHtml()` 每次生成 `<details class="evaluation-item" open>`，导致用户收起的评测卡片再次展开。
- `captureDetailState()` / `restoreDetailState()` 只保存 `#rolloutContent details` 的展开布尔数组；其他卡片不保留，数组按顺序恢复对增删/重排不稳。
- `loadRollouts()` 把 `rolloutReading` 置为 `true`，但完成后未正常复位；“正在请求”和“暂停阅读更新”两个状态被混用。
- 必须考虑异步响应乱序：切换实验后，旧请求不应覆盖新实验。

### 约定行为

1. **数据轮询继续进行**；未变化的实验详情不重绘，变化时尽量局部更新。
2. 为所有可展开卡片赋稳定 `data-ui-key`：如 `evaluation:<id>`、`stability:unavailable`、`group:<step>:<uid>`、`response:<step>:<uid>:<rollout_id>`；真实 ID 缺失时标记不确定并采用受控回退。
3. 按实验 ID 保存 `details.open`、Step、选项、图表范围、滚动位置；**手动刷新也不得强制展开**。
4. `rolloutLoading` 仅表示请求进行中，`finally` 复位；如有阅读暂停需求，另建 `livePaused` 或 `freezeDetail` 状态。
5. 为请求增加序号或 `AbortController`，过期响应不得覆盖当前页面；切换实验时复位当前实验无关状态。
6. Rollout 证据应改为 Group 级分页并展示总数、已加载数、是否完整；不要让接口的 100 条上限表现成“这一步全部记录”。

### 必须通过的浏览器验收

- 收起独立评测卡片后等待至少三次轮询，仍保持收起。
- Group A 展开、Group B 收起；刷新后状态不变。
- 手动刷新、切换实验 A→B→A、回看旧 Step、滚动阅读，状态符合预期。
- 新训练 Step 出现时后台数据更新，阅读中的详情不闪烁或跳顶。
- 快速多次切换实验、并发请求返回乱序，不显示旧实验。
- API 缺失/超时/部分 JSONL 尚未写完时可恢复，不能重置用户交互。
- Web 和 Jupyter 共享指标语义；Jupyter 不必复刻全部 Web 交互。

## 7. 架构与 UI 去冗余

建议保留现有 `ExperimentStore` 与报告能力，不推翻工程。

统一事实链：

```text
verl trainer + rollout JSONL + actor update evidence + independent evaluations
   ↓
标准解析与身份解析（唯一口径；保留缺失和来源）
   ↓
Step / Group / Response / Actor Update 聚合
   ↓
规则引擎 + 机制诊断 + 受控实验建议
   ↓
Web / Jupyter / HTML / RL-Insight 关联链接
```

建议 Web 以四个工作区组织，而不按算法名无限增加页面：

1. **Overview**：实验进度、独立评测摘要、重点异常和数据可信度。
2. **Rollout Explorer**：Prompt → Group → Response → Reward 分量 → Advantage → 更新证据。
3. **Training Dynamics**：Advantage、Entropy、KL、Clip/Ratio、Grad Norm、机制诊断卡与证据时间线。
4. **Experiment Comparison / Performance**：配置差异、独立效果、有效 Group / 生成 Token / Wall Time、RL-Insight 证据链接。

具体去重：合并“训练现象”与“稳定性判断”的规则来源；避免同一指标在 KPI、六图、最近 Step 表多处重复且定义不同；不同量纲的 Reward/比例、KL/Entropy 拆轴或拆图。引入 Step → Group → Response 的证据钻取，不单设“逐 Token Reward”页面。

## 8. 建议修改文件与开发顺序

| 里程碑 | 优先级 | 建议改动位置 | 可交付物 / 验收 |
|---|---|---|---|
| M0：修复现有缺陷 | P0 | `web/app.js`；`stability_analyzer.py`、`stability_config.py`、`experiment_rules.py` | 展开状态回归测试；Entropy/耗时方向测试；重复规则归并设计 |
| M1：统一事实源 | P1 | `live_monitor.py`、`rollout_parser.py`、`experiment_store.py`、`dashboard.py` | 同一数据两条入口指标一致；身份不明明确提示；Group 完整分页 |
| M2：GRPO/NGRPO 首期机制诊断 | P1 | `analyzers/group_signal.py`；新增 `algorithm_profiles.py`、`recommendation_rules.py` | 同 Reward+非零 NGRPO Advantage 不误报；建议包含证据与复验 |
| M3：Actor Update 证据 | P2 | 扩展训练侧可选埋点；新增 `analyzers/policy_update.py`；`schemas/data_models.py` | 非零优势、正负裁剪、Loss 分母与真实 Update 关联；缺失不伪算 |
| M4：算法机制对照 | P2 | `algorithm_profiles.py`、分析器、Web | DAPO/Dr.GRPO/GSPO 仅在真实机制和数据满足前提时计算 |
| M5：效果与性能复验 | P3 | `evaluation_store.py`、`docs/rl_insight/`、比较 UI | 同 seed/数据/预算的单变量对照；质量/吞吐/资源独立结论 |

### 第一阶段最小闭环（推荐先做）

在一条真实 GRPO 或 NGRPO 训练中：

1. 从异常 Step 定位到原始 Group 和 Response。
2. 展示真实 Reward、已记录 Advantage、非零优势 Token 候选数与来源。
3. 依据当前算法机制解释为何有/没有组内学习信号，不推断尚未采集的裁剪与真实梯度。
4. 给出条件性建议（检查 Reward/采样、或安排 GRPO↔NGRPO 对照），同时列出独立评测验收项。
5. 在 UI 自动刷新、切换实验及阅读展开卡片时，结果与用户状态不丢失。

### 测试和报告规范

- 单元测试覆盖：相同 Reward 但 NGRPO Advantage 非零；缺失 UID/Log Prob；奖励为负但正确性未知；算法未启用对应机制；Token Mask 边界；截断与 Clip 语义分离。
- 训练回归至少对比：GRPO、当前 NGRPO；同数据/seed/预算，评测 F1/Recall 等业务指标并记录 Reward、KL、Entropy、训练耗时。
- 每项优化分别记为：`OPTIMIZED` / `NO_BOTTLENECK` / `NO_IMPROVEMENT` / `BLOCKED` / `INCONCLUSIVE`；没有显著结果也保留证据。
- 文档明确区分：**源码已确认**、**论文机制借鉴**、**设计提议**、**实验已验证**；运行时性能及模型能力不能相互替代。

## 9. 明确不包含的范围

本方案**不**表示已经支持 GSPO、DAPO、Dr.GRPO、GAPO 的实际训练，也不表示 Token Reward 或 Actor Update 的逐 Token 梯度归因已经可用；不承诺自动根因定位、自动改超参、自动复训、准生产风控验收。上述内容若未来开发，应分别追加真实埋点、公式级单元测试与可复现对照。

## 10. 下一次实施的任务拆分建议

先创建三个独立 Issue/开发任务，而不是同时重构：

1. **[P0] Web 轮询时保留卡片展开/折叠及阅读状态**：直接以 §6 的浏览器验收为准。
2. **[P0] 修正规则方向与统一事件语义**：以 §2 的 Entropy、耗时、KL/梯度/截断重叠问题为准。
3. **[P1] GRPO/NGRPO 机制诊断 MVP**：先基于现有 `advantage` 等字段完成“现象 → 机制 → 建议 → 复验”，不新增大规模 Token Log Prob 落盘。

后续按照 M1–M5 扩展，并在每个阶段更新 [当前状态](CURRENT_STATUS_20261112.md) 记录“已完成/未完成”与对应真实验证证据。
