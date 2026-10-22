# 复现手册

## 1. 复现范围

Git 仓库可复现数据、Reward、SFT/RL 启动、Rollout 汇总、独立评测和报告生成。模型权重、镜像、NPU 驱动和历史 checkpoint 需在目标环境另行提供。

建议先记录：仓库 commit、镜像 digest、模型目录版本、NPU 驱动、CANN、vLLM/vLLM-Ascend 和 VERL revision。

## 2. 生成数据

```bash
python3 tools/amlsim/prepare_amlsim_typology.py \
  --output /path/to/amlsim_typology_rules_v1
```

验收：`manifest.json` 中 train/dev/test 为 2400/400/800，每类均衡，且六个 Parquet SHA-256 与 `results/dataset_manifest.json` 一致。哈希不一致时不要与已发布结果横向比较。

## 3. Base 评测

```bash
MODEL_HOST=/path/to/qwen3-8b \
DATA_HOST=/path/to/amlsim_typology_rules_v1/rl/test.parquet \
OUTPUT_HOST=/path/to/evaluations/E0-base-full \
  bash tools/amlsim/launch_amlsim_eval.sh
```

验收参考：格式有效率约 100%，Risk Recall 约 7.83%。数值明显不同应先检查模型 revision、chat template 和生成参数。

## 4. SFT

```bash
MODEL_HOST=/path/to/qwen3-8b \
DATA_HOST=/path/to/amlsim_typology_rules_v1/sft \
TOTAL_STEPS=30 \
  bash tools/amlsim/launch_amlsim_sft.sh
```

Launcher 会先应用并检查 EOS loss patch。训练完成后使用 step 30 的 `huggingface` 目录评测。验收参考：格式有效率约 99.25%，Risk Accuracy 约 89.75%。

## 5. GRPO

```bash
MODEL_HOST=/path/to/sft-step30/huggingface \
DATA_HOST=/path/to/amlsim_typology_rules_v1/rl \
ADV_ESTIMATOR=grpo TOTAL_STEPS=5 SAVE_FREQ=1 \
  bash tools/amlsim/launch_amlsim_rl.sh
```

训练后汇总：

```bash
python3 tools/amlsim/summarize_amlsim_rollouts.py \
  --rollouts /path/to/grpo-run/rollouts \
  --output /path/to/grpo-run/analysis
```

验收参考：step 5 exact rate 约 93.75%，gradient group rate 约 0%。

## 6. NGRPO

```bash
MODEL_HOST=/path/to/sft-step30/huggingface \
DATA_HOST=/path/to/amlsim_typology_rules_v1/rl \
ADV_ESTIMATOR=ngrpo TOTAL_STEPS=5 SAVE_FREQ=1 \
NGRPO_MAX_REWARD=4.0 \
  bash tools/amlsim/launch_amlsim_rl.sh
```

Launcher 在容器中应用 NGRPO patch。必须确认 patch check 成功，并确认 `NGRPO_MAX_REWARD` 等于 Reward 最大值 4.0。

## 7. 合并 checkpoint 与独立评测

```bash
python3 -m verl.model_merger merge --backend fsdp \
  --local_dir /path/to/checkpoint/actor \
  --target_dir /path/to/checkpoint/actor/merged_huggingface

MODEL_HOST=/path/to/merged_huggingface \
DATA_HOST=/path/to/amlsim_typology_rules_v1/rl/test.parquet \
OUTPUT_HOST=/path/to/evaluation \
  bash tools/amlsim/launch_amlsim_eval.sh
```

至少评测 GRPO step 5 和 NGRPO step 1/4/5。禁止用训练 Rollout exact rate 替代独立测试。

## 8. 生成报告

```bash
python3 tools/amlsim/build_amlsim_case_report.py \
  --evaluations /path/to/evaluations/amlsim \
  --grpo-run /path/to/grpo-run \
  --ngrpo-run /path/to/ngrpo-run \
  --output /path/to/case-study
```

输出 CSV、SVG、实验报告和 PPT 提纲。将新结果与 `docs/amlsim/results/` 比较时，应先核对数据哈希、模型、seed、参数和代码 revision。

## 9. 启动诊断 UI

```bash
rl-dashboard \
  --runs-root /path/to/runs \
  --state-dir /path/to/experiment_registry \
  --host 0.0.0.0 \
  --port 8765
```

UI 负责训练期观测与证据定位，不自动证明因果。独立评测完成后，再把 checkpoint 指标与训练内趋势共同写入结论。

## 10. 其他 agent 验收清单

- 数据哈希一致，或明确记录为何不一致。
- SFT EOS patch 与 NGRPO patch 均通过 check。
- GRPO/NGRPO 从同一 SFT checkpoint 启动。
- Reward 文件、最大值和版本一致。
- 每 step 的 Rollout、train.log、run_manifest 和 checkpoint 可对应。
- 独立测试使用同一 800 条样本、temperature 0。
- 同时报告总体指标和四类指标。
- 区分直接观测、代码定位与解释性推断。
- 不把合成规则阈值描述为银行生产规则。
