# RL Training Toolkit：跨任务强化学习基准与 Agent 长期执行计划

> 版本：v1.0 / 制定：2026-11-16  
> 类型：可分阶段交付的工程规范 + 实验协议 + Agent 接续手册  
> 作用范围：`sonneImmer/rl-training-toolkit`；本文件是**计划**，不是“功能已实现”或“新实验已跑完”的声明。  
> 配套入口：仓库根目录 [AGENTS.md](../AGENTS.md)；已有算法机制方案 [ALGORITHM_MECHANISM_DIAGNOSTICS_ROADMAP.md](ALGORITHM_MECHANISM_DIAGNOSTICS_ROADMAP.md)。

## 0. 核心目标与成功定义

不要只在项目自己合成的 AMLSim 风格规则集上获得接近满分的准确率；通过**金融、数学、代码**三类公开来源数据，在 **8 × Ascend 910B3 / Qwen3-8B / verl** 的已验证基础上，形成可重复的强化学习算法对照，同时验证诊断系统能否解释真实的学习信号与性能代价。

整个项目分为四个独立问题：

1. **任务效果**：RL 是否在独立验证/测试中优于同一起点的 Base、SFT？是否跨难度/类别/数据分布成立？
2. **机制证据**：Correctness、Reward、Advantage、Policy Update 是否一致？各算法真正解决什么瓶颈？
3. **计算效率**：有效学习信号的样本/Token/时间成本，以及 NPU 显存、吞吐和训练阶段耗时。
4. **工具可靠性**：不同数据集共用同一标准验证与诊断协议；无数据时不生成伪结论；UI 不因轮询丢失阅读状态。

成功不是某一组得分最高，而是**同一评测协议下，有对照、可复现、能解释失败原因**。所有结果分别标注 `observed`、`proposed`、`blocked`、`inconclusive`，不将算法推荐写成自动证实的因果结论。

## 1. 截至 2026-11-16 的事实基线（做新任务前必须复核）

基线仓库 `main` 最新已检查提交：`00a5ff8ea4aa88f5c47a7ff1589923acda58849b`，提交主题 “Archive long-run algorithm evaluation results”。**后续提交会让此处过时；不得以本文替代实时检查。**

### 已存在的可复用资产

- `tools/finqa/`：官方 FinQA 到 verl Parquet 的转换、可执行程序/答案/证据 Reward、SFT 与 GRPO/NGRPO 脚本；目前有冒烟记录，**不是完整公开基准算法对照**。官方数据清单参考 `docs/finqa/results/rl_dataset_manifest.json`。
- `tools/verified_math/`：数学输入预检与短程 GRPO/NGRPO 训练链路；`src/rl_diagnostics/evaluators/math_evaluator.py` 已支持 GSM8K/MATH 风格抽取、部分数值比较和 Pass@k，但**复杂代数/LaTeX 等价正确性尚不足**。
- `src/rl_diagnostics/evaluators/code_evaluator.py` 已有代码抽取、语法检查、测试、超时统计；**现有 Python 子进程 + 正则 API 检查不构成可信隔离沙箱，禁止直接用于执行任意不可信模型代码的正式训练**。
- `src/rl_diagnostics/analyzers/group_signal.py` 已有 Reward 差异、部分正确性分层、实际 Advantage 字段分析；`algorithm_profiles.py` 与 `recommendation_rules.py` 已支持 GRPO、虚拟最大值 NGRPO、PPO/GAE 的基础机制卡。
- `patch_tools/patch_verl_rollout_evidence.py`、`patch_verl_ngrpo.py`、`patch_verl_rl_insight*.py` 为训练证据、优势函数和 RL-Insight 埋点。实际要按镜像/verl 版本进行 patch check/restore 验证。
- 最新 `web/app.js` 已出现 `data-ui-key` 与展开状态保存逻辑；稳定性配置也已拆分 Entropy 与耗时/吞吐方向。**旧文档曾记录的问题可能已修复**：先做浏览器/单测回归，不能根据旧计划判定仍有缺陷。

### 当前 AMLSim 结果仅作为工具链 Sanity Check

`docs/amlsim/results/algorithm_comparison_20261112/README.md` 与 `comparison.md` 中，单 seed、150 步、固定 800 条**同生成规则分布**测试集上：GRPO 99.750%、NGRPO 99.875%、PPO 100.000% 风险准确率（误报分别为 2/1/0，漏报均为 0）。训练低信息 Group 比例约 97.54% / 98.25% / 95.00%。

这些结果证明本任务上的训练与算法路径，但**不能**证明真实金融场景泛化、PPO 普遍占优、无数据泄漏，或算法改善一定来自特定机制。保留归档，不以此集继续追求小数点后的准确率。

## 2. 公开数据集矩阵与用途

**优先按任务类型接入，不按数据集数目堆新 Reward 函数。** 数据集规模、许可、题目版本和官方 split 必须在下载时重新核对并写入 manifest。

| 任务线 | 数据集 | 目的 | 当前状态 / 下一步 |
|---|---|---|---|
| 金融数值推理 | **FinQA** | 第一条公开完整对照：SFT/GRPO/NGRPO/PPO，程序与答案验证 | **部分已实现**；完成可复验基准 |
| 金融表格+文本 | **TAT-QA** | 表格/文本、单位/运算与跨金融问答泛化 | **计划**；新 Dataset Adapter + 校验器组合 |
| 金融多轮推理 | ConvFinQA | 多轮上下文与长度分布压力测试 | **扩展**；不抢占第一阶段资源 |
| 金融外部评测 | FinanceBench | 真实财报问答的额外独立评测 | **评测候选**，先核对许可与文档访问，绝不当作训练集直接混入 |
| 信贷违约 | UCI Default of Credit Card Clients | 真实信用卡客户历史违约分类与传统模型比较 | **计划**；另建分类任务与成本敏感评测 |
| 企业破产 | UCI Taiwanese Bankruptcy Prediction | 既有风控原型，多 seed、独立分布与分类指标复验 | **已有链路**；不得与 UCI Credit Default 混称一个数据集 |
| 数学基础 | **GSM8K** | 数据/答案提取与 Reward 冒烟；高分时不扩充训练 | **解析器已有**；需官方导入 |
| 数学正式 | **MATH** | 分难度/分题型、不同算法的相对信号与长度效应 | **评测器部分已有**；需可靠数学等价验证 |
| 数学保留评测 | **MATH-500** | 独立数学评测 | **评测候选**；与 MATH 题目去重，严格排除训练/调参泄漏 |
| 数学采样研究 | DAPO-Math-17k | 有效组采样、Clip-Higher、Token-Level Loss 研究 | **候选**；必须固定精确数据版本和来源 |
| 代码基础 | **MBPP** | 安全执行、单测 Reward 冒烟 | **评测器部分已有**；安全沙箱是硬前置 |
| 代码正式 | **APPS** | Introductory / Interview 分难度的程序正确性与效率 | **计划**；适配函数调用/标准输入输出任务及测试规范 |
| 代码保留评测 | LiveCodeBench | 固定版本与时间切分的泛化检查 | **评测候选**；评估公开测试和潜在预训练污染 |

公开来源入口（真实接入前由 Agent 验证链接、许可、版本及哈希）：
- FinQA：`https://github.com/czyssrs/FinQA`；TAT-QA：`https://github.com/NExTplusplus/TAT-QA`；ConvFinQA：`https://github.com/czyssrs/ConvFinQA`。
- UCI Credit Default：`https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients`；Taiwanese Bankruptcy：使用仓库现有 `docs/finance_risk/` 数据溯源。
- GSM8K：`https://huggingface.co/datasets/openai/gsm8k`；MATH：`https://github.com/hendrycks/math`；MATH-500：`https://huggingface.co/datasets/HuggingFaceH4/MATH-500`。
- DAPO-Math：`https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k`。
- MBPP：`https://github.com/google-research/google-research/tree/master/mbpp`；APPS：`https://huggingface.co/datasets/codeparrot/apps`；LiveCodeBench：`https://github.com/LiveCodeBench/LiveCodeBench`。
- FinanceBench：`https://huggingface.co/datasets/PatronusAI/financebench`。

### 选择下一数据集的规则

1. 完整对照尚未建立：**FinQA 优先**（少做重复接入工程）。
2. 数学解析器/验证器可通过黄金测试：**GSM8K 冒烟 → MATH 主实验 → MATH-500 保留评测**。
3. 安全代码执行平台未验收：**禁止 MBPP/APPS 的真实模型代码批量执行**；仅做接口及合成良性单测。
4. 完成沙箱门禁后：**MBPP 冒烟 → APPS 正式实验 → LiveCodeBench 独立评测**。
5. 金融分类必须含传统基线：信用违约/破产任务另设 Logistic Regression、LightGBM/XGBoost 对照，按标签类计算 Precision/Recall/F2/PR-AUC 与成本指标。
6. ConvFinQA、TAT-QA、FinanceBench、Elliptic 等只在主任务基准可靠且资源足够时拓展。

## 3. 统一架构：Dataset Adapter → Task Verifier → Reward Policy → Diagnostics

重用现有 `evaluators/`、`analyzers/` 与 `tools/<task>/`，**优先抽取公共接口，不一次性重写现有训练脚本**。

```text
原始数据（官方 split + 固定版本/哈希）
    ↓ Dataset Adapter：字段/题目/Prompt/答案结构/输出协议
CanonicalSample
    ↓ 模型 Rollout（带 dataset_id、sample_id、group_uid、step、attempt）
    ↓ Task Verifier：客观检验结果（不关心 GRPO/NGRPO/PPO）
VerificationResult
    ↓ Reward Policy：把验证事实映射为具体训练分数/分量
RewardResult
    ↓ verl：优势、Loss、Actor 更新 + 证据落盘
    ↓ 标准 Rollout/Group/Step 解析器
    ↓ 通用 + 算法特有的诊断规则
    ↓ Web/Jupyter/HTML/独立评测/性能对照
```

建议新增/演进目录（**目标结构，不表示已存在**）：

```text
src/rl_diagnostics/
  datasets/              # base.py + gsm8k/math/finqa/mbpp/apps/... adapters
  verifiers/             # 通用契约 + math_equivalence/code_execution/finance
  rewards/               # outcome/partial_credit/composite 评分策略
  evaluators/            # 现有模块保留；作为评测入口和适配代理
  analyzers/group_signal.py
  algorithm_profiles.py
  recommendation_rules.py
```

### 复用准则

| 差别类型 | 做法 |
|---|---|
| `question`/`problem`、`answer` 字段名不同 | 用数据集映射配置，不复制验证器 |
| `#### 42` 与 `\\boxed{42}` 输出格式不同 | 共用数学验证流程，使用 Dataset-specific Answer Extractor |
| 数值容差或格式要求不同 | 同一个 Verifier + 明确的配置/版本 |
| 有符号等价、单位换算或计算程序 | 组合专门 Verifier；不能用纯字符串比较冒充等价 |
| MBPP 函数测试与 APPS stdin/stdout 测试 | 共用安全 Code Runner 和评分接口，增加任务 Runner Adapter |
| Reward 二元/部分得分/多分量 | **只改变 Reward Policy**，不修改数据集解析或客观正确性 |
| 任务根本无可靠标准答案 | `is_correct=null`，不计算正确性混合率；可评估 Reward 差异 |

### 数据契约（接口草案）

```python
@dataclass
class VerificationResult:
    sample_id: str
    status: str        # correct | incorrect | parse_error | verifier_error | blocked
    is_correct: bool | None
    score: float | None = None  # 可选连续验证成绩，不是训练 Reward
    components: dict = field(default_factory=dict)
    verifier_id: str = ""
    verifier_version: str = ""
    error_type: str | None = None

@dataclass
class RewardResult:
    reward: float | None
    components: dict
    policy_id: str
    policy_version: str
    min_reward: float | None
    max_reward: float | None
    status: str        # scored | unscorable | blocked
```

- `correct`/`incorrect` 来自独立、完整的 Verifier 定义；模型不能仅凭 Reward 大小宣称正确。
- `verifier_error`/`blocked` 是**验证未知**，不得当成错误样本计入正确性分母，也不应静默给 0 Reward 训练。
- 任务格式错误可被单独定义为错误，但须事先声明规则；训练 Reward 失败的异常路径必须可见。
- `verifier_version`、`reward_policy_version`、`dataset_sha256`、`prompt_template_version`、`model/tokenizer`、`tool_version` 与 `seed` 必须写入实验 manifest。
- NGRPO `max_reward` 必须与该 Reward Policy **可达到的真实上界**吻合；改变 Reward 范围时阻止错误配置或显式报错。
- 所有可选数据集字段应记录缺失状态，不能通过语言模型猜补 Ground Truth。

### 数学与代码验证的特别门禁

**数学**：现有 `math_evaluator.py` 数值比较功能可复用，但正式 MATH 需支持可靠的数学表达式/符号等价、异常解析与分歧审计；准备至少“等价不同写法、非等价近似、单位差异、无法解析”黄金样例，并对误判案例人工复核。不要使用易误判的字符串相等替代完整数学验证。

**代码**：现有 `code_evaluator.py` 的 `subprocess`、`resource.setrlimit` 和危险 API 正则检查**不具备安全边界**。必须先使用专门隔离 Worker/容器或微虚拟机，禁止网络、宿主文件/密钥访问，限制权限、系统调用、CPU/内存/进程/时间；沙箱失败时返回 `blocked`，不能降级为直接执行。运行训练单测与保留评测用例需严格隔离；对多测试程序返回逐测试结果、部分通过率、编译失败、运行错误、超时原因。禁止运行不可信模型代码于有模型权重和金融数据挂载的训练宿主机。

## 4. 通用 Group 诊断：Correctness ≠ Reward ≠ Advantage

**适用于一切具备可靠逐 Response 正误标签的任务；不需要每个数据集重复实现。**

对有至少两个有效响应、且每条均有明确布尔正确性标签的组 `g`，设 `c_{g,i} ∈ {0,1}`：

- Mixed Correctness：`0 < Σ_i c_{g,i} < G_g`。
- `Mixed Correctness Group Rate = 混合正确性 Group 数 / 标签完整且组身份可靠的 Group 数`。
- **All-correct / All-wrong** 单独统计；缺失验证、UID 不确定、样本未齐全时标记 `not_evaluated`，不得默认填零。
- **Reward Informative Group Rate**：按固定浮点容差/已声明 Reward 离散精度判断组内评分存在有意义差异的比例。
- **Nonzero Advantage Group Rate**：依赖**实际训练中记录的** Advantage；不能用 Reward 反推 NGRPO/PPO 更新。
- **Reward–Correctness Agreement**：检查高评分与真实验证结果是否一致；Reward 为连续分量时须先定义可比较的具体判据，不得直接以 `reward>0` 定义 correctness。
- **Partial Credit**：代码四个 Rollout 均未全过测试，测试通过率为 [0.2,0.4,0.7,0.8] 时，混合正确性 = false，但 Reward 仍可能有组内区别。

对于 4 条独立同分布的二元 Rollout，仅作帮助理解的理论参考：`P(mixed) = 1 - p^4 - (1-p)^4`；实际数据不一定独立同分布，不应用公式替代观测统计。

建议所有通用指标输出：`value`、`numerator`、`denominator`、`coverage`、`status`、`source`、`group_id_quality`、`verifier_version`。Web 的“有效 Group”需要标明究竟是正确性混合、Reward 有区别，还是非零 Advantage，不能混合成一个百分比。

## 5. Reward Policy 的复用与逐任务边界

| 模板 | 适用 | 核心验证 | Reward 策略建议 |
|---|---|---|---|
| `math_outcome` | GSM8K、MATH | 答案抽取 + 数值/代数等价 | 默认客观答案 0/1；格式仅作为单独分量，防 Reward Hacking |
| `program_execution` | MBPP、APPS、HumanEval（评测） | 独立安全执行+单元测试 | 全过 0/1 和部分通过率是**两种对照策略**，不得混淆 correctness |
| `financial_numeric` | FinQA、TAT-QA、ConvFinQA | 程序或表格数值推理、单位、证据 | 重用计算/答案核验；FinQA 程序执行与 TAT-QA 证据格式需子适配 |
| `risk_classification` | Taiwanese Bankruptcy、UCI Credit Default | 标签、正负类定义、阈值/成本 | 固定标签语义，独立报告 Recall/Precision/F2/PR-AUC；可对比二元和成本敏感策略 |
| `evidence_qa_eval` | FinanceBench（保留评测） | 引用、答案证据 | 不直接当训练 Reward 入口，先解决评测许可和事实核查 |

严格比较：固定验证器与 Reward 版本；若研究不同 Reward 策略，列为**Reward 消融实验**，不能把 Reward 变化与算法切换同时发生后声称算法更优。所有 Reward 要有边界、失败模式、标注稽核样本和专家检查结果。

## 6. 算法矩阵：先现有算法，后针对现象引入新机制

**第一阶段：每个主任务先比较 Base、SFT、GRPO、当前实现的 NGRPO（虚拟最高奖励版本）和 PPO/GAE。** 使用同一个**该任务的 SFT checkpoint**、相同数据/Prompt/主要训练预算，记录 Critic 等特有额外计算。首次接入可先跑 GRPO/NGRPO，PPO 按资源扩展。

| 方法/机制 | 在何种证据下研究 | 必须注意 |
|---|---|---|
| GRPO | 同质 Reward / 零 Advantage / 难度不匹配 | Group 训练信号不等于泛化质量 |
| 本仓库 NGRPO 虚拟最高 Reward | 全错同质组出现负优势时 | 全同高分可能仍为零；负优势不保证帮助训练；勿视作包含所有论文 NGRPO 机制 |
| PPO/GAE | Critic/GAE 对稀疏奖励可能保持非零优势 | 额外 Critic 开销单独报告；不能用 GRPO 的混合组率等同 PPO 有效性 |
| DAPO Dynamic Sampling | **确有**生成 Token/重试/过滤后低信号成本时 | 全错组对 NGRPO 可能仍有学习价值；不能统一过滤 |
| DAPO Clip-Higher | 有 Actor Update 内正/负优势方向裁剪证据 | 需实际 Ratio/Clip 分支，而非仅 `pg_clipfrac` |
| DAPO Token-Level Loss / Dr.GRPO | 已知长度分布 + 实际 Loss 归一化下权重偏斜 | 要从实际 Loss Reduction 中记录分母，不凭文本长度猜测梯度 |
| GSPO | Token 级 Ratio 不稳定，并能采到 Sequence Ratio | GSPO 用**序列级**重要性比率/裁剪，不是 Token Reward |
| GAPO | 已明确具体同名论文/机制版本 | 不把同名研究混用；未确定时不实现“通用 GAPO” |

仅在 Actor Update 阶段有来源明确的旧/新 Log Prob、Mask、裁剪边界、Reduction 与 Update ID 时，才能展示 Token/Sequence 更新证据。只存在 `policy_signal_tokens` 时仅表明非零优势候选 Token 数，不宣称真实梯度。未来引入逐步骤/逐 Token 过程奖励，必须有可靠过程验证器，不从最终答案评分虚构 Token Reward。

## 7. 8 卡 910B3 受控实验协议

硬件和 Qwen3-8B 是当前**建议主基线**，并非无需预检就能运行所有长序列或 PPO Critic 配置。

**执行前约束：**
- 每个任务都保存独立的 Base/SFT Checkpoint 基线。绝不能用 FinQA 训练后的 checkpoint 作为 MATH、APPS 的“同起点算法比较”，除非明确设定为跨任务迁移实验。
- 官方 train 训练、dev 验证调参、test **最终一次性报告**；保留集、测试样例/程序、不泄漏到 Prompt、Reward 训练和迭代选择。MATH-500 与 MATH 可能有重合，须跨 split 按题目/答案指纹去重后才能称独立。
- 先做输入和训练前检查：数据许可、行数/hash、标签语义、Prompt 是否泄漏答案、格式错误、Verifier 的黄金样例和 Reward 上限、NPU 内存估计、离线环境依赖。
- 每个数据集固定一个初版协议：模型/Tokenizer/Prompt、`rollout.n=4`、seed=42、Batch/最大输入输出长度和 KL 等参数。`150 steps` 只是实验起点，不是跨任务必须遵守的训练定额。
- 建议 `Step 0 / 50 / 100 / 150` 检查点（显存/磁盘允许情况下）；优先完成单 seed 正确性验证，再用 42/43/44 三个 seed 复验有希望的方案。
- 公平性分别按：**相同数据/采样预算/更新步数**比较训练信号；**相同总生成 Token 或 NPU Wall Time**比较实际效率。PPO 额外 Critic、代码 CPU 单测消耗必须计入。
- 评测每个 checkpoint 时使用固定独立数据和确定解码，按任务报告主指标和分层结果；数学包括 exact/equivalence accuracy、parse error、Pass@k；代码包括 full test pass、pass@1/Pass@k、编译/超时/隔离阻断、单测覆盖；金融还需 Recall、Precision、F2、混淆矩阵、证据一致性和适用的传统模型基线。
- 保留每次运行 `run_manifest.json`、`dataset_manifest.json`、`verifier/reward_version`、`train.log`、`rollouts/`、`evaluations/`、`diagnosis.json`、`timing/`、`case_report.md`；Git 只提交代码、脚本、匿名化小型汇总、哈希和报告，不上传完整模型、权重、原始金融样本或密钥。
- 调参顺序：先看 Base/SFT 在训练难度区间是否有可学习 Group；若过难/过易，先分层和调整数据，不把算法切换当成唯一解决方案。Reward 上升但独立评测无改善时先检查 Reward–Correctness 一致性。

## 8. 分阶段执行任务（Agent 每次只领取当前最小可验证任务）

状态定义：`TODO` / `IN_PROGRESS` / `BLOCKED` / `DONE_VERIFIED` / `INCONCLUSIVE`。只允许有测试/证据路径的任务标成 `DONE_VERIFIED`。本节是首次安排，执行前必须查最新代码和日志更新现状。

### M0：事实审计与回归（优先）
- [ ] 核对最新版日志解析、UI 刷新/折叠、Entropy/耗时规则、Prompt Group 归属是否已有验证覆盖；**已修代码不重复重写**。
- [ ] 固定 AMLSim 与 FinQA 既有实验的不可变来源、模型/Reward/Verifier 版本，展示合成数据接近饱和但不推导算法排名。
- [ ] 建立通用诊断的黄金输入样例：全对/全错/混合、部分评分、缺 UID、验证失败、NGRPO 全错非零 Advantage、PPO GAE 同质 Reward。
- **验收**：同一记录在实时/离线/网页中含义一致；旧实验依然可读取；失败证据和缺失数据被明确标记。

### M1：共用数据与验证契约
- [ ] 提炼 `CanonicalSample`、`VerificationResult`、`RewardResult` 和版本化 manifest；首批以 FinQA 和 GSM8K 两条线覆盖复用。
- [ ] 在 `group_signal.py` 中统一正确性混合率、Reward 区分率、Advantage 非零率，分母与未知状态有单测；不新增第二套 Group 统计。
- [ ] 训练 Reward 与独立评测器分离，建立数据集适配器注册/映射；历史字段不强制破坏兼容。
- **验收**：相同验证事实切换 Outcome/Partial Reward 时 `is_correct` 不变；未知验证不会变成错误或零 Reward；NGRPO 上限不合法时显式报错。

### M2：FinQA 公开完整对照
- [ ] 复用 `tools/finqa/`，核对官方 train/dev/test、程序 Reward、答案与证据、来源哈希。
- [ ] 完成 SFT 基线/GRPO/NGRPO；PPO 在 Critic 路径预检后追加；固定独立评测并生成机制诊断卡。
- [ ] 先用 1-step smoke，确认训练数据、Reward、日志、保存与评测正常；后续训练需资源确认。
- **验收**：同起点、同协议、真实指标、失败与开销均入档；无新训练则保持 TODO，不伪造结果。

### M3：数学推理线
- [ ] GSM8K 的 `####` 数据适配 + 数值黄金样例。
- [ ] MATH 的 `\boxed{}`、LaTeX/表达式等价验证黄金样例、难度/题型分层；MATH-500 严格留作独立评测并检查重合。
- [ ] GRPO/NGRPO baseline；根据真实长度权重/低信息组证据才安排 Dr.GRPO 或 DAPO 单变量实验。
- **验收**：验证器正确性审计、跨分层独立指标、实际非零 Advantage/同质组分析、训练和评测资源账本。

### M4：代码线（安全硬门禁）
- [ ] 替换/封装现有不安全子进程运行器；隔离执行、网络/文件权限/资源限制，恶意输入/逃逸预防的安全审查。
- [ ] MBPP 函数测试任务适配与良性测试预检；APPS stdin/stdout 与多测试适配；记录全过和部分通过两种奖励方案。
- [ ] 沙箱单独验收通过后，才可开放模型生成代码的真实执行及 RL；LiveCodeBench 仅保留评测。
- **验收**：沙箱缺失必须 `blocked` 且训练不执行；完整/部分测试成绩和训练 Reward 映射一致；异常终止不能影响宿主机。

### M5：金融业务泛化线
- [ ] TAT-QA（表格+文本）在共享数值验证器基础上新增文档/单位/证据 Adapter。
- [ ] UCI Credit Default 与 Taiwanese Bankruptcy 分开建模，固定类别分布/阈值，并加入传统模型基线。
- [ ] ConvFinQA/FinanceBench 仅在先验主任务通过后作为扩展。
- **验收**：不能把 FinQA 指标当作信贷风控指标；真实分布与合成分布分开汇报；记录类别召回和错误成本。

### M6：算法机制与 RL-Insight 对照
- [ ] 在当前算法机制卡上增加**数据足够才显示**的建议；先 Group/Advantage，再正负裁剪、长度权重、Sequence Ratio。
- [ ] 新算法/新技巧一次只改一个变量，对照 GRPO/NGRPO/PPO 的独立质量、有效学习信号、NPU/CPU 计算成本。
- [ ] RL-Insight 用唯一 experiment ID/Step/Worker 关联生成、训练、同步及评测；对重复实验检查噪声。
- **验收**：每条建议包含观察值、机制、适用条件、缺失证据和复验标准；“性能更快”和“效果更好”分别报告。

## 9. Agent 的持续执行协议（跨会话/重启）

后续 Agent 必须遵守**有界、自校验、可接力**工作，而不是未经授权后台反复训练。

1. **读现状**：读取根目录 `AGENTS.md`、本文件、`README.md`、`docs/CURRENT_STATUS_20261112.md`、最近 commits 和关联脚本；确认默认分支、Git 工作树、正在发生的其他修改。旧日期文档只作历史证据。
2. **选择一个最小任务**：按 M0→M6 门禁，选择依赖满足且一轮能完成/验证的最小 Task；若已完成则通过代码与测试核实后跳过，不重复实现。
3. **先声明**：任务 ID、输入/输出、将改文件、可用环境与风险；如果需要占用 8 NPU、下载受限数据、暴露网络、使用大量外部费用，先获得明确授权。
4. **执行**：先单测/小样本 smoke 再真实运行；不覆盖用户已有 checkpoint/实验结果，不更改默认 Reward/算法语义而不标版本。
5. **验证**：记录实际执行命令、退出码、指标来源、异常与未完成项。无法执行 NPU 时，仅提交离线代码/黄金测试与 `BLOCKED` 原因，不得写“训练完成”。
6. **记录交接**：本文件 `10 进度日志` 追加一条可追溯记录；与代码同一次提交，注明 commit、测试、下一项最小任务。不要把过时的计划“TODO”机械复制为现状。
7. **风险处理**：敏感数据/凭据泄漏、沙箱安全边界不成立、官方测试集污染、无法复现实验时停止相应路径，记录 `BLOCKED` 并转做其他无资源前置任务。
8. **结束条件**：每轮完成一个可验证交付物后停止，不自动跨多阶段消耗资源；持续性来自代码+版本化日志+后续 Agent 接续，不是声称后台自动运行。

### Agent 最小交付格式

```text
Task ID / 目标：
读取的基线提交与数据版本：
修改文件与设计决策：
测试命令与实际结果：
训练/评测证据路径（若没有，写“未执行”）：
风险/阻塞/不可推断之处：
状态：DONE_VERIFIED / BLOCKED / INCONCLUSIVE / ...
下一最小任务：
提交 commit：
```

## 10. 进度日志（只追加真实事实；任务状态发生变化需更新）

| 日期 | 任务 ID | 状态 | 证据/提交 | 下一步 |
|---|---|---|---|---|
| 2026-11-16 | PLAN-001 | DONE_VERIFIED（文档） | 本文件首次提交；无新基准训练 | M0 核实已有修复与测试后选择 M1 |

> 第 10 节的完成只表示**计划文档已提交**；M0–M6 并未因此实现。更新时保留历史行和实验失败记录。

## 11. 长期维护禁止事项

- 禁止将训练 Reward、独立评测正确率、Group 混合率、非零 Advantage、Token Loss 权重、真实参数梯度混称一个指标。
- 禁止未核验就下载/训练官方测试集；不把训练公开单测与保留隐藏测试混为一谈；不在保留测试集上循环选超参数。
- 禁止将合成 AMLSim 近满分结果解释为真实反洗钱性能或算法普遍优越。
- 禁止仅从 `adv_estimator` 算法名称推断完整实现了 GSPO/DAPO/NGRPO 全套机制。
- 禁止把未经隔离的模型生成代码交给宿主 Python 执行。
- 禁止提交敏感金融数据、完整模型/权重、访问令牌、内网绝对路径及不应公开的完整 Rollout。
- 禁止未检查最新源码就重修旧 bug 或重建已经存在的功能。
- 禁止凭空填写未执行训练的跑分、算力节省或“已自动修复”结论。

## 12. 与已有文档的关系

- [算法机制诊断与 UI 方案](ALGORITHM_MECHANISM_DIAGNOSTICS_ROADMAP.md)：更详细的算法诊断证据、前端交互与 Actor Update 埋点设计；其历史缺陷状态必须以最新代码复核。
- [当前旧版状态总览](CURRENT_STATUS_20261112.md)：2026-11-12 的历史事实切片，不覆盖之后新提交。
- [AMLSim 三算法对照](amlsim/results/algorithm_comparison_20261112/README.md)：最新已归档的合成规则集结果与局限。
- [FinQA 链路](finqa/README.md)：真实可复用的金融问答导入和 Reward 入口。
- [环境恢复](../environment/RECOVERY.md)：8 卡环境、镜像与外部资产恢复依赖。
