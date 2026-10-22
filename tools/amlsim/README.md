# AMLSim rule-based RL experiment

This directory reproduces the AML risk, typology, and evidence experiment used by the diagnostics dashboard.

## Scope

The dataset generator follows the public IBM AMLSim directional semantics for `FAN_IN`, `FAN_OUT`, and `CYCLE`, and adds deliberately broken near-patterns as `NORMAL` controls. It does not execute the original AMLSim Java simulator. The four-account thresholds, 48/72-hour windows, negative controls, class balance, prompt contract, and reward weights are choices made for this controlled experiment; they are not presented as AMLSim defaults or bank rules.

The complete audit and reproduction package is under `docs/amlsim/`.

The expected model output is exactly three lines:

```text
Risk: SUSPICIOUS|NORMAL
Typology: FAN_IN|FAN_OUT|CYCLE|NORMAL
Evidence: comma-separated transaction IDs or NONE
```

The reward range is `[-4, 4]`: risk classification contributes 2 points, typology contributes 1 point, and transformed evidence multiset F1 contributes from -1 to 1 point. Invalid three-line format receives -4.

## Reproduction

Generate the controlled dataset:

```bash
python3 tools/amlsim/prepare_amlsim_typology.py \
  --output /mnt/local/transformer_files/zkx/data/amlsim_typology_rules_v1_20261021
```

Run the SFT stage. The launcher applies the VERL EOS-loss patch at container startup:

```bash
TOTAL_STEPS=30 bash tools/amlsim/launch_amlsim_sft.sh
```

Run matched GRPO and NGRPO stages from the same SFT checkpoint:

```bash
MODEL_HOST=/path/to/sft/huggingface ADV_ESTIMATOR=grpo TOTAL_STEPS=5 SAVE_FREQ=1 \
  bash tools/amlsim/launch_amlsim_rl.sh

MODEL_HOST=/path/to/sft/huggingface ADV_ESTIMATOR=ngrpo TOTAL_STEPS=5 SAVE_FREQ=1 \
  bash tools/amlsim/launch_amlsim_rl.sh
```

Summarize group and typology signals:

```bash
python3 tools/amlsim/summarize_amlsim_rollouts.py \
  --rollouts /path/to/run/rollouts --output /path/to/run/analysis
```

Merge a VERL FSDP actor checkpoint before evaluation:

```bash
python3 -m verl.model_merger merge --backend fsdp \
  --local_dir /path/to/checkpoint/actor \
  --target_dir /path/to/checkpoint/actor/merged_huggingface
```

Evaluate the same independent split for every checkpoint:

```bash
MODEL_HOST=/path/to/merged_huggingface OUTPUT_HOST=/path/to/evaluation \
  bash tools/amlsim/launch_amlsim_eval.sh
```

Build the case report and PPT source charts:

```bash
python3 tools/amlsim/build_amlsim_case_report.py \
  --evaluations /path/to/evaluations/amlsim \
  --grpo-run /path/to/grpo-run \
  --ngrpo-run /path/to/ngrpo-run \
  --output /path/to/case-study
```

Published metrics and charts are snapshots of the 2026-10-21 run. Regenerate the dataset and compare the SHA-256 values in `docs/amlsim/results/dataset_manifest.json` before rerunning training. Large artifacts such as model weights, Parquet data, complete predictions, and logs remain outside Git.
