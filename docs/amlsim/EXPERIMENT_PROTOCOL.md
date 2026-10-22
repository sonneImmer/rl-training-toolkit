# 实验协议

## 1. 研究问题

- RQ1：Base 模型的失败来自格式问题，还是无法识别交易结构？
- RQ2：SFT 能否先完成严格输出对齐，并保留风险与类型识别能力？
- RQ3：GRPO 训练中是否出现 Reward 漏洞、全对/全错 group 增多和相对学习信号消失？
- RQ4：在相同 SFT checkpoint、数据、采样和预算下，NGRPO 是否改善独立测试效果？
- RQ5：总体指标是否会掩盖单一业务类型退化？

## 2. 预注册式预期

| 阶段 | 实验前预期 | 失败判据 |
| --- | --- | --- |
| Base | 能遵循三行格式，但结构识别较弱 | 格式有效率低，无法区分能力与格式 |
| SFT | 格式率显著提高，风险/类型优于 Base | loss 下降但生成不终止或独立指标不升 |
| GRPO | Reward、exact match、独立指标继续改善 | Reward 上升但独立测试不升；有效 group 快速归零 |
| NGRPO | 虚拟满分样本使同分但低于满分的 group 获得非零负优势 | 仅总体风险提高但类型宏观或单类明显下降 |

NGRPO 不是预设赢家。算法判断必须以同一独立测试集为准，训练 Reward 和小批次 Rollout 仅用于诊断。

## 3. 计算环境

- 设备：8 张 Ascend 910B3 NPU。
- 镜像：`openpangu-72b-2512-postrain:v1-zkx-pangu-mathfix`。
- 镜像记录 ID：`d2a7a76de939-derived`。
- Base 模型：Qwen3-8B，本地挂载到容器 `/mnt/model`。
- VERL 代码路径：镜像内 `/workspace/verl`。
- NGRPO：容器启动时用 `patch_tools/patch_verl_ngrpo.py` 运行时打补丁，不重建镜像。
- SFT：容器启动时用 `tools/amlsim/patch_verl_sft_eos.py` 修复 EOS loss mask。
- 实验代码 revision：运行清单记录为 `dcf791335687fb2b0591497b485733fafe66644c`；本次文档化提交是后续 revision。

## 4. SFT 参数

| 参数 | 值 |
| --- | --- |
| 训练步数 | 30（有效重跑） |
| train/dev | 2400 / 400 |
| global batch | 16 |
| micro batch / NPU | 1 |
| 最大序列长度 | 2048 |
| 学习率 | `1e-5` |
| seed | 42 |
| NPU | 8 |
| strategy | FSDP，CPU offload |
| 目标 | 严格三行 Risk/Typology/Evidence |

初始错误 SFT 运行了 80 步，用作故障分析，不作为 RL 起点。有效 RL 起点为 EOS 修复后 step 30 的 Hugging Face checkpoint。

## 5. GRPO/NGRPO 共同参数

| 参数 | 值 |
| --- | --- |
| 训练步数 | 5 |
| train/test | 2400 / 800 |
| train batch | 16 prompts |
| rollout n | 4 responses / prompt |
| 每 step Rollout | 64 responses，16 groups |
| response length | 160 |
| actor LR | `5e-7` |
| PPO mini batch | 8 |
| micro batch / NPU | 1 |
| temperature / top_p | 1.0 / 0.95 |
| tensor parallel | 4 |
| KL loss coefficient | 0.001 |
| KL type | `low_var_kl` |
| FSDP | parameter 与 optimizer offload |
| seed | 42 |
| checkpoint frequency | 每 step（本轮实际设置） |

NGRPO 额外参数：`algorithm.max_reward=4.0`、`algorithm.add_max_reward_num=1`。实现会在每组统计量中加入一个虚拟满分 4.0；当组内真实样本同分且低于 4.0 时，可产生非零负优势。若全组真实样本已经都是 4.0，优势仍为 0，因此它不能消除所有信号饱和。

## 6. 独立评测

所有 checkpoint 在同一 800 条 test split 上，以温度 0、最大生成 160 token 评测。指标包括：Risk confusion matrix、Accuracy、Precision、Recall、F1、F2；Typology 总准确率、混淆矩阵、每类准确率；严格三行格式有效率；Evidence 样本平均 Precision、Recall、F1。

Base、SFT、GRPO step 5、NGRPO step 1/4/5 均使用相同评测脚本 `tools/amlsim/evaluate_amlsim.py`。

## 7. 因果控制与未控制因素

已控制：相同 SFT 起点、数据集、seed、训练步数、batch、Rollout 采样、Reward、硬件数量和评测集。

未完全控制：NPU/vLLM 运行可能存在非确定性；只有单个训练 seed；没有重复运行置信区间；GRPO/NGRPO 只训练 5 步；合成数据不代表生产分布。因此结果是案例级证据，不是算法普适性结论。
