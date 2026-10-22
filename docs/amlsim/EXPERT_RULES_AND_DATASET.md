# 专家规则与数据生成来源

## 1. 来源边界

上游来源为 IBM AMLSim，仓库为 `https://github.com/IBM/AMLSim.git`，本实验锁定 revision `7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca`，许可证为 Apache-2.0。

AMLSim 源码提供了三类交易拓扑的方向性语义：

| 类型 | 上游语义 | 上游证据位置 |
| --- | --- | --- |
| FAN_IN | 多个 originator 向一个 main beneficiary 转账 | `src/main/java/amlsim/model/aml/FanInTypology.java`；`scripts/transaction_graph_generator.py` |
| FAN_OUT | 一个 main originator 向多个 beneficiary 分发 | `src/main/java/amlsim/model/aml/FanOutTypology.java`；`scripts/transaction_graph_generator.py` |
| CYCLE | 多账户首尾相接形成有向闭环 | `src/main/java/amlsim/model/aml/CycleTypology.java`；`scripts/transaction_graph_generator.py` |

本实验没有运行 AMLSim Java 仿真器生成完整银行网络，而是编写了确定性受控生成器 `tools/amlsim/prepare_amlsim_typology.py`。因此准确表述是“基于 AMLSim 拓扑语义的合成数据”，不是“AMLSim 官方数据集”或“真实银行数据”。

## 2. 本实验定义的专家规则

| 标签 | 判定规则 | 正例证据 |
| --- | --- | --- |
| FAN_IN | 48 小时内至少 4 个不同源账户向同一中心账户转入 | 构成完整汇入结构的全部交易 ID |
| FAN_OUT | 48 小时内同一中心账户向至少 4 个不同目标账户转出 | 构成完整分散结构的全部交易 ID |
| CYCLE | 72 小时内至少 4 个账户构成闭合有向转账环 | 构成完整环的全部交易 ID |
| NORMAL | 不完整满足上述任一规则；包括故意打断条件的近似模式 | `NONE` |

这些规则在数据生成之前固定。标签、Risk 和 Evidence 均由生成器中的结构事实直接给出，不使用专家模型、教师模型或训练中 AI 评分。

必须区分两类“专家经验”：

- 上游公共语义：资金汇聚、资金分散、闭环转账的结构定义来自 AMLSim。
- 本轮实验参数化：最少 4 个对手方、48/72 小时时窗、负例构造、类别平衡和 Reward 权重由项目组为受控验证设定，不声称是 AMLSim 默认值或银行生产规则。

## 3. 数据生成过程

1. 为每个样本创建独立账户命名空间，避免 train/dev/test 账户泄漏。
2. 正例先注入完整 FAN_IN、FAN_OUT 或 CYCLE 结构，再加入与结构账户隔离的噪声交易。
3. NORMAL 从 `broken_fan_in`、`broken_fan_out`、`broken_cycle`、`ordinary` 中采样。近似模式至少破坏时间窗、闭环或结构完整性之一。
4. 交易顺序打乱，Prompt 中再按时间和交易 ID 排序展示。
5. 同时生成 RL Parquet 与 SFT Parquet；两者共享 `sample_id` 和同一事实标签。
6. 输出固定为三行：`Risk`、`Typology`、`Evidence`。本数据不包含自由文本 CoT。

## 4. 数据规模与确定性

随机种子为 42。train/dev/test 分别使用种子 42/43/44，case ID 起点分别为 0/100000/200000。

| Split | 总量 | 每类数量 | RL SHA-256 | SFT SHA-256 |
| --- | ---: | ---: | --- | --- |
| train | 2400 | 600 | `c4761c1be0c90cd9bd5e2d7f6b78c4d24d24fe32a7f91f9dd74823c1b7a6b846` | `a6fb121188f977582b09d970c82ec87ca4e07486ad36f4a54aeb381bfef4a028` |
| dev | 400 | 100 | `7314eec6cd073865e7aa8c3367128aafcc0daeb8b2c6bfa4c746acda35cd5028` | `7afb57b633dcd94296f6f7c1b11c5d50992b65fb934ed756cac07d87b6f831db` |
| test | 800 | 200 | `1acb3214fb3d912b8cfac64ddacac0a27ec51ee73fd45d9b0937f9f5c9cc6b1d` | `9d23bc8d767b2c3a7586f98117de2a6a1293226d390476b0b53b88d1ac27451d` |

复现时必须由脚本重新计算哈希，并与 `results/dataset_manifest.json` 对比。任何哈希不一致都应停止后续横向比较。

## 5. Reward 合约

Reward 范围为 `[-4, 4]`：

- Risk 正确 `+2`，错误 `-2`。
- Typology 正确 `+1`，错误 `-1`。
- Evidence 得分为 `2 * multiset_F1 - 1`，范围 `[-1, 1]`。
- 输出不是严格三行格式时，总分直接设为 `-4`。

Evidence 使用多重集合 F1。重复输出同一交易 ID 会增加预测集合分母，从而被惩罚；这是在首次 GRPO 暴露 Reward 漏洞后修复的规则。

## 6. 适用边界

该数据适合验证结构化规则遵循、证据选择、Reward 设计、Rollout 分组诊断和算法对比。它不适合直接估计真实银行反洗钱效果，原因包括：类别均衡不符合真实发生率、规则确定性较高、账户行为缺少长期历史、没有跨账户实体关系噪声，也没有经过合规专家验证的生产阈值。
