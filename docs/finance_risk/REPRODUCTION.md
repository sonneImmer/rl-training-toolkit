# 金融风控复现说明

## 数据

使用 `tools/finance_risk/prepare_taiwanese_bankruptcy.py` 从 UCI 数据压缩包生成固定 train/dev/test：

```bash
python3 tools/finance_risk/prepare_taiwanese_bankruptcy.py \
  --source-zip /path/to/taiwanese_bankruptcy_prediction.zip \
  --output /path/to/finance_taiwanese_bankruptcy_v2
```

复现时核对 `results/dataset_manifest.json` 的源文件哈希、split 哈希、36 个字段和随机种子。train 使用 1:3 下采样，dev/test 保留自然分布。

## SFT

先生成 SFT 数据，再启动训练：

```bash
python3 tools/finance_risk/prepare_finance_risk_sft.py \
  --source /path/to/finance_taiwanese_bankruptcy_v2 \
  --output /path/to/finance_taiwanese_bankruptcy_sft

MODEL_HOST=/path/to/qwen3-8b DATA_HOST=/path/to/sft-data \
  bash tools/finance_risk/launch_finance_sft.sh
```

Launcher 会在容器内应用 VERL EOS loss patch。SFT 只保证输出协议学习，必须在固定测试集独立评测分类能力。

## GRPO/NGRPO

```bash
MODEL_HOST=/path/to/sft-checkpoint DATA_HOST=/path/to/rl-data \
ADV_ESTIMATOR=grpo TOTAL_STEPS=5 SAVE_FREQ=1 \
  bash tools/finance_risk/launch_finance_grpo.sh

MODEL_HOST=/path/to/sft-checkpoint DATA_HOST=/path/to/rl-data \
ADV_ESTIMATOR=ngrpo TOTAL_STEPS=5 SAVE_FREQ=1 NGRPO_MAX_REWARD=1.0 \
  bash tools/finance_risk/launch_finance_grpo.sh
```

纯结果 Reward `credit_risk_outcome_v1` 的范围是 `[0, 1]`。切换类别加权 Reward 时必须同时设置正确的 Reward 文件、规则 ID 和最大/最小值。

## 汇总与评测

```bash
python3 tools/finance_risk/summarize_finance_rollouts.py /path/to/rollouts \
  --output /path/to/rollout_summary.json

MODEL_HOST=/path/to/merged-checkpoint \
DATA_HOST=/path/to/test.parquet OUTPUT_HOST=/path/to/evaluation \
  bash tools/finance_risk/launch_finance_eval.sh
```

必须报告 Precision、Recall、F1、F2、FN、FP、格式有效率和混淆矩阵。自然分布测试集不得用于训练采样或 Reward 调参。
