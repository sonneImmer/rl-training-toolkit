#!/usr/bin/env bash
set -Eeuo pipefail

: "${MODEL_PATH:=/mnt/model}"
: "${DATA_DIR:=/mnt/data}"
: "${RUN_ROOT:=/mnt/runs/finqa-grpo-$(date +%Y%m%d)}"
: "${TOTAL_STEPS:=5}"
: "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3}"
: "${IMAGE_NAME:=unknown}"
: "${CODE_REVISION:=unknown}"
: "${RANDOM_SEED:=42}"
: "${ADV_ESTIMATOR:=grpo}"

TOOLS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$RUN_ROOT"
RUN_DIR=$(mktemp -d "$RUN_ROOT/finqa-$ADV_ESTIMATOR.XXXXXXXX")
export PYTHONPATH="$TOOLS_DIR${PYTHONPATH:+:$PYTHONPATH}"
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
export DEVICE_COUNT MODEL_PATH DATA_DIR TOTAL_STEPS IMAGE_NAME CODE_REVISION RANDOM_SEED ADV_ESTIMATOR

python3 "$TOOLS_DIR/preflight_finqa.py" \
  --data "$DATA_DIR" --model "$MODEL_PATH" --reward "$TOOLS_DIR/finqa_reward.py" \
  --output "$RUN_DIR/preflight" --max-prompt-length 4096

python3 - "$RUN_DIR/run_manifest.json" <<'PY_MANIFEST'
import json, os, sys
from pathlib import Path
manifest = {
    "schema_version": "1.0.0",
    "task": {"domain": "finance", "type": "financial_numerical_reasoning"},
    "dataset": {"id": os.environ["DATA_DIR"], "feature_count": "financial_text_table"},
    "prompt": {"contains_cot": False, "reasoning_format": "executable_program"},
    "reward": {"definition": "finqa_execution_answer_evidence_v2", "min": -4.0, "max": 4.0},
    "model": {"id": os.environ["MODEL_PATH"]},
    "training": {
        "algorithm": os.environ["ADV_ESTIMATOR"],
        "seed": int(os.environ["RANDOM_SEED"]),
        "total_steps": int(os.environ["TOTAL_STEPS"]),
    },
    "resources": {
        "npu_count": int(os.environ["DEVICE_COUNT"]),
        "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(","),
    },
    "environment": {
        "image": os.environ["IMAGE_NAME"],
        "patch_revision": "finqa-executable-v2",
        "code_revision": os.environ["CODE_REVISION"],
    },
}
Path(sys.argv[1]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
PY_MANIFEST

ALGORITHM_ARGS=("algorithm.adv_estimator=$ADV_ESTIMATOR")
if [[ "$ADV_ESTIMATOR" == "ngrpo" ]]; then
  ALGORITHM_ARGS+=("algorithm.max_reward=4.0" "algorithm.add_max_reward_num=1")
fi

python3 -m verl.trainer.main_ppo \
  "${ALGORITHM_ARGS[@]}" \
  data.train_files="$DATA_DIR/train.parquet" data.val_files="$DATA_DIR/dev.parquet" \
  data.train_batch_size=8 data.val_batch_size=8 data.max_prompt_length=4096 data.max_response_length=256 \
  data.filter_overlong_prompts=True data.truncation=error data.shuffle=True data.seed="$RANDOM_SEED" \
  custom_reward_function.path="$TOOLS_DIR/finqa_reward.py" custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.actor.ppo_mini_batch_size=4 \
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
  actor_rollout_ref.rollout.agent.default_agent_loop=finqa_single_turn \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$TOOLS_DIR/finqa_agent_loop.yaml" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 trainer.val_before_train=False trainer.logger=console \
  trainer.project_name=finqa_executable_reasoning trainer.experiment_name="$(basename "$RUN_DIR")" \
  trainer.n_gpus_per_node="$DEVICE_COUNT" trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_epochs=1 trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$RUN_DIR/checkpoints" trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  2>&1 | tee "$RUN_DIR/train.log"
