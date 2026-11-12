#!/usr/bin/env bash
set -Eeuo pipefail

: "${MODEL_PATH:=/mnt/model}"
: "${DATA_DIR:=/mnt/data}"
: "${RUN_ROOT:=/mnt/runs/rl-insight-memory-$(date +%Y%m%d)}"
: "${TOTAL_STEPS:=3}"
: "${TRAIN_BATCH_SIZE:=64}"
: "${SAVE_FREQ:=999}"
: "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${GRAD_SYNC_MODE:=per_micro_batch}"
: "${PPO_MINI_BATCH_SIZE:=64}"
: "${PPO_MICRO_BATCH_SIZE:=1}"
: "${RANDOM_SEED:=42}"

if [[ "$GRAD_SYNC_MODE" != "deferred" && "$GRAD_SYNC_MODE" != "per_micro_batch" ]]; then
  echo "GRAD_SYNC_MODE must be deferred or per_micro_batch" >&2
  exit 1
fi

AMLSIM_TOOLS=/mnt/amlsim_tools
mkdir -p "$RUN_ROOT"
RUN_DIR=$(mktemp -d "$RUN_ROOT/${GRAD_SYNC_MODE}.XXXXXXXX")
export RL_INSIGHT_EXPERIMENT="$(basename "$RUN_DIR")"
STARTED_AT=$(date -Is)
export RUN_DIR STARTED_AT GRAD_SYNC_MODE RANDOM_SEED TOTAL_STEPS TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE

finish_case() {
  status=$?
  FINISHED_AT=$(date -Is) STATUS=$status python3 - "$RUN_DIR/case_timing.json" <<'PY'
import json, os, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ["FINISHED_AT"],
    "exit_code": int(os.environ["STATUS"]),
    "gradient_sync_mode": os.environ["GRAD_SYNC_MODE"],
}, indent=2) + "\n")
PY
  exit "$status"
}
trap finish_case EXIT

export VLLM_ASCEND_ENABLE_NZ=0 HYDRA_FULL_ERROR=1 HCCL_CONNECT_TIMEOUT=600
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-vlan699}" HCCL_WHITELIST_DISABLE=1
export VERL_GRAD_SYNC_MODE="$GRAD_SYNC_MODE"
IFS=',' read -ra DEVICE_LIST <<< "$ASCEND_RT_VISIBLE_DEVICES"
DEVICE_COUNT=${#DEVICE_LIST[@]}
TENSOR_PARALLEL_SIZE=4
if [[ "$DEVICE_COUNT" -lt 4 ]]; then TENSOR_PARALLEL_SIZE="$DEVICE_COUNT"; fi

python3 "$AMLSIM_TOOLS/preflight_amlsim.py" \
  --data "$DATA_DIR" --model "$MODEL_PATH" --reward "$AMLSIM_TOOLS/amlsim_reward.py" \
  --output "$RUN_DIR/preflight" --max-prompt-length 2048

python3 - "$RUN_DIR/run_manifest.json" <<'PY'
import json, os, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "1.0.0",
    "task": {"domain": "rl_insight_memory_case", "type": "fsdp_gradient_sync_memory"},
    "training": {
        "algorithm": "grpo",
        "seed": int(os.environ["RANDOM_SEED"]),
        "total_steps": int(os.environ["TOTAL_STEPS"]),
        "train_batch_size": int(os.environ["TRAIN_BATCH_SIZE"]),
        "gradient_sync_mode": os.environ["GRAD_SYNC_MODE"],
        "ppo_mini_batch_size": int(os.environ["PPO_MINI_BATCH_SIZE"]),
        "ppo_micro_batch_size_per_gpu": int(os.environ["PPO_MICRO_BATCH_SIZE"]),
    },
    "monitoring": {
        "stack": "rl-insight-0.3.0",
        "npu_exporter": "172.60.6.4:19101",
        "prometheus": "172.60.6.4:19090",
        "rl_insight_server": "172.60.6.4:18080",
    },
    "experiment_note": "deferred mode is an intentionally injected regression for the controlled case study",
}, ensure_ascii=False, indent=2) + "\n")
PY

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_DIR/train.parquet" data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" data.val_batch_size=16 data.max_prompt_length=2048 data.max_response_length=160 \
  data.filter_overlong_prompts=True data.truncation=error data.shuffle=True data.seed="$RANDOM_SEED" \
  custom_reward_function.path="$AMLSIM_TOOLS/amlsim_reward.py" custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.disable_log_stats=False actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_PARALLEL_SIZE" actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 trainer.val_before_train=False trainer.logger=console \
  trainer.project_name=rl_insight_memory_case trainer.experiment_name="$(basename "$RUN_DIR")" \
  trainer.n_gpus_per_node="$DEVICE_COUNT" trainer.nnodes=1 trainer.save_freq="$SAVE_FREQ" trainer.test_freq=-1 \
  trainer.total_epochs=100 trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$RUN_DIR/checkpoints" trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  2>&1 | tee "$RUN_DIR/train.log"
