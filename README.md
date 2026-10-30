# RL Training Toolkit

面向 VERL 强化学习训练的可复用补丁与可视化工具。

- `patch_tools/`：在运行中容器内给 VERL 增加 NGRPO，支持应用、检查和恢复。
- `patches/`：让 rollout 输出包含稳定分组信息的最小静态补丁。
- `src/rl_diagnostics/`：解析日志和 rollout，分析组内信号、回答质量与训练稳定性，并输出 HTML/SVG。
- `notebooks/`：适用于只有 Jupyter/终端环境的按 step 实时监控页面。

本仓库不保存模型、数据集、训练日志、完整 rollout 原文或检查点。为保证案例可审计，`docs/amlsim/results/` 保存经过标准化的小型指标快照、运行清单和汇报图表；原始大文件由确定性脚本、数据哈希和复现手册定位。

AMLSim 反洗钱案例入口：`docs/amlsim/README.md`。

## 安装

```bash
python3 -m pip install -e .
```

开发测试：

```bash
python3 -m pip install -e '.[dev]'
pytest -q
```

## Rollout 证据补丁

新训练在启动 VERL 前应用 rollout 证据补丁，可为 GRPO 和 NGRPO 同时保存 `uid`、算法名以及每条 Response 的序列级 `advantage`：

```bash
python3 patch_tools/patch_verl_rollout_evidence.py apply \
  --verl-root /workspace/verl \
  --backup-dir /workspace/verl_rollout_evidence_backup

python3 patch_tools/patch_verl_rollout_evidence.py check \
  --verl-root /workspace/verl
```

`advantage` 是进入 Policy Loss 的相对学习方向，不等同于该 Group 的实际梯度。历史 rollout 没有该字段时，UI 显示“未记录”，不会根据 Reward 反推。仓库中的 RL 启动脚本已经自动应用此补丁。

新版证据补丁还记录：

- `selected_for_update`：该 Response 是否进入调用 `update_actor` 的批次；
- `response_mask_tokens`：参与 Actor loss mask 的 Response token 数；
- `policy_signal_tokens`：Advantage 非零且未被 Response mask 屏蔽的 token 数。

`policy_signal_tokens=0` 表示没有策略梯度信号，但开启 KL loss 时仍可能存在 KL 梯度，因此不能解释为“完全没有任何梯度”。Reward 函数返回的 `reward_components` 会按原始结构保留，并在 UI 中展示各分量是否具有组内区分度。

## NGRPO 运行时补丁

对于 `openpangu-72b-2512-postrain:v1`，推荐直接应用基于该镜像原始 VERL 文件生成的完整补丁：

```bash
patch --batch --forward --directory=/workspace/verl -p1 \
  < patches/002-ngrpo-algorithm.patch
```

该补丁一次修改 NGRPO 算法实现、算法配置类和默认 YAML，不修改 `ray_trainer.py`、vLLM 或可视化代码。

Python patcher仍可用于需要备份和恢复的环境：

```bash
python3 patch_tools/patch_verl_ngrpo.py apply \
  --verl-root /workspace/verl \
  --backup-dir /workspace/verl_ngrpo_backup

python3 patch_tools/patch_verl_ngrpo.py check \
  --verl-root /workspace/verl
```

训练时增加：

```text
algorithm.adv_estimator=ngrpo
algorithm.max_reward=1.0
algorithm.add_max_reward_num=1
```

`max_reward` 必须等于奖励函数能产生的实际最高分。恢复原文件：

```bash
python3 patch_tools/patch_verl_ngrpo.py restore \
  --backup-dir /workspace/verl_ngrpo_backup
```

## 离线诊断与图表

```bash
rl-diagnostics analyze \
  --log /path/to/train.log \
  --rollout-dir /path/to/rollouts \
  --output-dir /path/to/diagnostics_output \
  --export-svgs
```

输出包含结构化 CSV/JSON、HTML 报告和可用于汇报的 SVG 图表。输入与输出目录默认由 `.gitignore` 排除。

训练期只记录可复现事实：reward 分布、组内是否有学习信号、训练指标，以及训练产物中明确提供的 `acc/pred`。阈值用于筛选“接近全正/全负”或“接近全对/全错”的组，不生成经验难度，也不自动解释原因。原始数据没有 `difficulty` 时直接跳过该维度。

## Jupyter 实时监控

打开 `notebooks/rl_rollout_live_monitor.ipynb`，设置 `RUNS_ROOT` 和仓库外的 `STATE_DIR` 后运行。Notebook 展示实验目录、标准 rollout 质量、组内学习信号、可展开稳定性规则、训练趋势、原始 rollout 证据，以及两个实验之间的参数、最新指标和对齐趋势。只依赖 Python 标准库与 Jupyter 自带的 IPython。

## 实验对象与前端看板

正式看板将每个包含 `train.log` 的运行目录视为一个实验对象。训练过程中定时更新，结束后保留最终快照、标签事件和产物路径。状态目录应放在 Git 仓库之外：

稳定性规则分为两级：少于正式窗口但已有至少 3 个 step 时显示“早期正常/早期预警（低置信度）”；达到规则要求的基线和观察窗口后，自动切换为正式判断。用户读取 Rollout 证据时，后台仍继续采集，但前端不会自动重建当前详情；手动点击“刷新”可恢复最新详情。

```bash
rl-dashboard \
  --runs-root /mnt/local/transformer_files/zkx/runs \
  --state-dir /mnt/local/transformer_files/zkx/experiment_registry \
  --host 0.0.0.0 \
  --port 8765
```

浏览器打开 `http://<服务器地址>:8765`。看板只读取训练日志和 rollout，不控制训练进程；标签仅描述可验证现象，并保存规则版本、阈值和证据 step。

新实验应在运行目录写入 `run_manifest.json`。`tools/verified_math/run_verified_math.sh` 会自动生成该文件，可通过环境变量补充实验事实：

```bash
export DOMAIN=finance_risk
export TASK_TYPE=risk_classification
export FEATURE_COUNT=36
export CONTAINS_COT=true
export REWARD_DEFINITION=finance_risk_v1
export REWARD_MIN=-4
export REWARD_MAX=4
export RANDOM_SEED=42
export IMAGE_NAME=openpangu-72b-2512-postrain:v1
export PATCH_REVISION=ngrpo-patch-v1
export CODE_REVISION=$(git rev-parse HEAD)
```

看板使用清单展示任务、Reward、资源和版本，并检查实验是否具备严格算法对比所需的配置记录。旧实验没有清单时显示“未记录”，不会猜测缺失信息。

## 独立评测接入

独立测试完成后，将 `metrics.json` 挂到对应训练运行目录：

```bash
python3 tools/attach_evaluation.py \
  --run-dir /path/to/run \
  --metrics /path/to/evaluation/metrics.json \
  --evaluation-id checkpoint-step-5 \
  --checkpoint-step 5 \
  --split test \
  --dataset dataset_version
```

看板会同时展示训练 Rollout 与固定测试集结果。若相邻 checkpoint 的总体指标上升、但某个类别准确率下降至少 5 个百分点，会触发“总体上升但单类下降”告警。该功能只读取已有评测结果，不负责合并或运行 checkpoint。

## 实验材料导出与本地查看

实验归档只包含看板需要的 `train.log`、`run_manifest.json`、rollout JSONL 和诊断/报告材料，不包含 checkpoint 或模型权重。导出全部实验：

```bash
python3 tools/experiment_bundle.py export \
  --runs-root /mnt/local/transformer_files/zkx/runs \
  --output /mnt/local/transformer_files/zkx/exports/rl-experiments.tar.gz \
  --all
```

只导出指定实验时，可重复提供相对 `runs-root` 的 `--run`：

```bash
python3 tools/experiment_bundle.py export \
  --runs-root /mnt/local/transformer_files/zkx/runs \
  --output /tmp/amlsim-case.tar.gz \
  --run amlsim-rl-20261021/amlsim-rl.QkHeuNtm
```

将归档下载到本地电脑，克隆本仓库后导入并启动：

```bash
python3 tools/experiment_bundle.py import \
  --archive ~/Downloads/rl-experiments.tar.gz \
  --output ~/rl-experiment-runs
python3 -m pip install -e .
rl-dashboard \
  --runs-root ~/rl-experiment-runs \
  --state-dir ~/rl-experiment-state \
  --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。归档内的 `manifest.json` 记录工具 Git revision、文件清单与 SHA256；导入时自动校验。`experiment_registry` 是可重建缓存，无需导出。

## 已归档实验案例

- `docs/ARCHIVE_INDEX.md`：归档范围和状态约定。
- `docs/finance_risk/`：金融风控 RL 原型案例与复现材料。
- `docs/finqa/`：FinQA 可执行 Reward 基准工具链。
- `docs/amlsim/`：AMLSim 反洗钱 RL 案例与完整实验结果。
