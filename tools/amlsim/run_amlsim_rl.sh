#!/usr/bin/env bash
set -Eeuo pipefail

: "${MODEL_PATH:=/mnt/model}"
: "${DATA_DIR:=/mnt/data}"
: "${RUN_ROOT:=/mnt/runs/amlsim-rl-$(date +%Y%m%d)}"
: "${TOTAL_STEPS:=5}"
: "${SAVE_FREQ:=$TOTAL_STEPS}"
: "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${IMAGE_NAME:=openpangu-72b-2512-postrain:v1-zkx-pangu-mathfix}"
: "${CODE_REVISION:=unknown}"
: "${RANDOM_SEED:=42}"
: "${ADV_ESTIMATOR:=grpo}"
: "${REWARD_FILE:=amlsim_reward.py}"
: "${REWARD_RULE_ID:=amlsim_risk_typology_evidence_v1}"
: "${REWARD_MIN:=-4.0}"
: "${REWARD_MAX:=4.0}"
: "${NGRPO_MAX_REWARD:=$REWARD_MAX}"
if [[ "$ADV_ESTIMATOR" != "grpo" && "$ADV_ESTIMATOR" != "ngrpo" ]]; then
  echo "ADV_ESTIMATOR must be grpo or ngrpo" >&2
  exit 1
fi

TOOLS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$RUN_ROOT"
RUN_DIR=$(mktemp -d "$RUN_ROOT/amlsim-rl.XXXXXXXX")
export VLLM_ASCEND_ENABLE_NZ=0
export HYDRA_FULL_ERROR=1
export HCCL_CONNECT_TIMEOUT=600
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-vlan699}"
export HCCL_WHITELIST_DISABLE=1
export ASCEND_RT_VISIBLE_DEVICES
IFS=',' read -ra DEVICE_LIST <<< "$ASCEND_RT_VISIBLE_DEVICES"
DEVICE_COUNT=${#DEVICE_LIST[@]}
if [[ "$DEVICE_COUNT" -ne 2 && "$DEVICE_COUNT" -ne 4 && "$DEVICE_COUNT" -ne 8 ]]; then
  echo "Supported NPU counts: 2, 4, 8; got $DEVICE_COUNT" >&2
  exit 1
fi
TENSOR_PARALLEL_SIZE=4
if [[ "$DEVICE_COUNT" -lt 4 ]]; then
  TENSOR_PARALLEL_SIZE="$DEVICE_COUNT"
fi
export DEVICE_COUNT TENSOR_PARALLEL_SIZE MODEL_PATH DATA_DIR TOTAL_STEPS SAVE_FREQ IMAGE_NAME CODE_REVISION RANDOM_SEED ADV_ESTIMATOR REWARD_FILE REWARD_RULE_ID REWARD_MIN REWARD_MAX NGRPO_MAX_REWARD

python3 "$TOOLS_DIR/preflight_amlsim.py" \
  --data "$DATA_DIR" --model "$MODEL_PATH" \
  --reward "$TOOLS_DIR/$REWARD_FILE" \
  --output "$RUN_DIR/preflight" --max-prompt-length 2048

python3 - "$RUN_DIR/run_manifest.json" <<'PY_MANIFEST'
import json, os, sys
from pathlib import Path
manifest = {
    "schema_version": "1.0.0",
    "task": {"domain": "amlsim", "type": "aml_risk_typology_evidence"},
    "dataset": {"id": os.environ.get("DATA_DIR", "/mnt/data"), "typologies": ["FAN_IN", "FAN_OUT", "CYCLE", "NORMAL"]},
    "prompt": {"contains_cot": False},
    "reward": {"definition": os.environ.get("REWARD_RULE_ID"), "min": float(os.environ.get("REWARD_MIN", "0")), "max": float(os.environ.get("REWARD_MAX", "1"))},
    "model": {"id": os.environ.get("MODEL_PATH", "/mnt/model")},
    "training": {"algorithm": os.environ.get("ADV_ESTIMATOR", "grpo"), "seed": int(os.environ.get("RANDOM_SEED", "42")), "total_steps": int(os.environ.get("TOTAL_STEPS", "5"))},
    "resources": {"npu_count": int(os.environ["DEVICE_COUNT"]), "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")},
    "environment": {"image": os.environ.get("IMAGE_NAME", "unknown"), "image_id": "d2a7a76de939-derived", "patch_revision": "rollout-json-mathfix", "code_revision": os.environ.get("CODE_REVISION", "unknown")},
}
Path(sys.argv[1]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
PY_MANIFEST

ALGORITHM_ARGS=("algorithm.adv_estimator=$ADV_ESTIMATOR")
if [[ "$ADV_ESTIMATOR" == "ngrpo" ]]; then
  ALGORITHM_ARGS+=("algorithm.max_reward=$NGRPO_MAX_REWARD" "algorithm.add_max_reward_num=1")
fi

python3 -m verl.trainer.main_ppo \
  "${ALGORITHM_ARGS[@]}" \
  data.train_files="$DATA_DIR/train.parquet" data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size=16 data.val_batch_size=16 data.max_prompt_length=2048 data.max_response_length=160 \
  data.filter_overlong_prompts=True data.truncation=error data.shuffle=True data.seed="$RANDOM_SEED" \
  custom_reward_function.path="$TOOLS_DIR/$REWARD_FILE" custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_PARALLEL_SIZE" actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 trainer.val_before_train=False trainer.logger=console \
  trainer.project_name=amlsim_verification trainer.experiment_name="$(basename "$RUN_DIR")" \
  trainer.n_gpus_per_node="$DEVICE_COUNT" trainer.nnodes=1 trainer.save_freq="$SAVE_FREQ" trainer.test_freq=-1 \
  trainer.total_epochs=100 trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$RUN_DIR/checkpoints" trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  2>&1 | tee "$RUN_DIR/train.log"
