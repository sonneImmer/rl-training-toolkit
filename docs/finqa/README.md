# FinQA 可执行 Reward 工具链

本目录归档 FinQA 金融数值推理的探索性工具链，用于验证“标准答案 + 可执行程序 + 证据引用”的确定性 Reward、SFT 格式对齐和 GRPO Rollout 接入。

## 已完成

- 将 FinQA 官方 train/dev/test 转换为 VERL Parquet。
- 实现无额外依赖的程序解析器和执行器。
- Reward 同时检查程序可执行性、执行结果、最终答案一致性和证据引用。
- 提供 SFT、GRPO/NGRPO 启动脚本、预检脚本和 agent loop 终止符。
- 数据清单和一次 GRPO 冒烟运行清单保存在 `results/`。

## 边界

FinQA 是财报数值推理，不是企业违约或反洗钱风控数据。现阶段只作为可执行 Reward 与训练链路基准，没有形成最终业务案例，也不应与 Taiwanese Bankruptcy 指标合并。

## 快速复现

```bash
python3 tools/finqa/validate_finqa.py --source /path/to/FinQA
python3 tools/finqa/prepare_finqa.py --source /path/to/FinQA --output /path/to/finqa_rl
python3 tools/finqa/prepare_finqa_sft.py --source /path/to/FinQA --output /path/to/finqa_sft

MODEL_HOST=/path/to/qwen3-8b DATA_HOST=/path/to/finqa_sft \
  bash tools/finqa/launch_finqa_sft.sh

MODEL_HOST=/path/to/sft-checkpoint DATA_HOST=/path/to/finqa_rl \
ADV_ESTIMATOR=grpo TOTAL_STEPS=5 \
  bash tools/finqa/launch_finqa_grpo.sh
```

NGRPO 启动器会自动应用运行时 patch；Reward 最大值必须保持为 4.0。
